import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

try:
    import clip
except ImportError:
    raise ImportError("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")

class ImageProjector(nn.Module):    
    def __init__(self, input_dim: int, output_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TextProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        output_dim: int = 256
    ):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CLIPFeatureExtractor(nn.Module):
    def __init__(self, clip_model_name: str = "RN101", device: str = "cuda"):
        super().__init__()
        
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=device)
        self.clip_model.eval()
        
        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.device = device
        self.model_name = clip_model_name
        self.is_resnet = "RN" in clip_model_name
        
        self._setup_dimensions()

    def train(self, mode: bool = True):
            super().train(mode)
            self.clip_model.eval()
            return self

    def _setup_dimensions(self):
        """Setup feature dimensions."""
        dim_map = {
            "RN50": (2048, 1024, 1024),
            "RN101": (2048, 2048, 512),
            "RN50x4": (2560, 2560, 640),
            "RN50x16": (3072, 3072, 768),
            "RN50x64": (4096, 4096, 1024),
            "ViT-B/32": (768, 768, 512),
            "ViT-B/16": (768, 768, 512),
            "ViT-L/14": (1024, 1024, 768),
        }
        self.layer4_dim, self.embed_dim, self.output_dim = dim_map.get(
            self.model_name, (2048, 2048, 512)
        )
        self.visual_dim = self.output_dim
        self.text_dim = self.output_dim
    
    def encode_image_local(self, images: torch.Tensor) -> torch.Tensor:
        if self.is_resnet:
            return self._encode_resnet_local(images)
        else:
            return self._encode_vit_local(images)
    
    def _encode_resnet_local(self, images: torch.Tensor) -> torch.Tensor:

        visual = self.clip_model.visual
        
        def stem(x):
            if hasattr(visual, 'relu1'):
                x = visual.relu1(visual.bn1(visual.conv1(x)))
                x = visual.relu2(visual.bn2(visual.conv2(x)))
                x = visual.relu3(visual.bn3(visual.conv3(x)))
            else:
                for conv, bn in [(visual.conv1, visual.bn1), (visual.conv2, visual.bn2), (visual.conv3, visual.bn3)]:
                    x = visual.relu(bn(conv(x)))
            x = visual.avgpool(x)
            return x
        
        x = images.type(visual.conv1.weight.dtype)
        x = stem(x)
        x = visual.layer1(x)
        x = visual.layer2(x)
        x = visual.layer3(x)
        x = visual.layer4(x)
        
        B, C, H, W = x.shape
        
        attnpool = visual.attnpool
        
        x = x.reshape(B, C, H * W).permute(2, 0, 1)
        
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        
        pos_embed = attnpool.positional_embedding 
        
        if pos_embed.shape[0] != H * W + 1:
            cls_pos = pos_embed[:1] 
            spatial_pos = pos_embed[1:] 
            
            orig_size = int(spatial_pos.shape[0] ** 0.5)
            spatial_pos = spatial_pos.permute(1, 0).reshape(1, -1, orig_size, orig_size)
            spatial_pos = F.interpolate(spatial_pos, size=(H, W), mode='bicubic', align_corners=False)
            spatial_pos = spatial_pos.reshape(-1, H * W).permute(1, 0) 
            
            pos_embed = torch.cat([cls_pos, spatial_pos], dim=0)  
        
        x = x + pos_embed[:, None, :].to(x.dtype) 
        
        x = F.linear(x, attnpool.v_proj.weight, attnpool.v_proj.bias)
        x = F.linear(x, attnpool.c_proj.weight, attnpool.c_proj.bias)
        
        x = x[1:]
        
        x = x.permute(1, 0, 2)  
        
        return x
    
    def _encode_vit_local(self, images: torch.Tensor) -> torch.Tensor:
        visual = self.clip_model.visual
        
        x = images.type(visual.conv1.weight.dtype)
        
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)
        
        x = x[:, 1:, :]
        
        if visual.proj is not None:
            x = x @ visual.proj
        
        return x
    
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_text(text_tokens)


class DCLIP(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "RN101",
        num_classes: int = 20,
        class_names: List[str] = None,
        proj_dim: int = 256,
        text_hidden_dim: int = 384,
        positive_prompt: str = "A photo of a {}.",
        negative_prompt: str = "A photo without a {}.",
        aggregation_scale: float = 5.0,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.device = device
        self.aggregation_scale = aggregation_scale
        
        self.clip = CLIPFeatureExtractor(clip_model_name, device)
        
        self.image_projector = ImageProjector(
            input_dim=self.clip.visual_dim,
            output_dim=proj_dim
        )
        
        self.text_projector = TextProjector(
            input_dim=self.clip.text_dim,
            hidden_dim=text_hidden_dim,
            output_dim=proj_dim
        )
        
        self.class_names = class_names if class_names else [f"class_{i}" for i in range(num_classes)]
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        
        self._precompute_text_features()
    
    def _precompute_text_features(self):
        with torch.no_grad():
            pos_texts = [self.positive_prompt.format(name) for name in self.class_names]
            pos_tokens = clip.tokenize(pos_texts).to(self.device)
            pos_features = self.clip.encode_text(pos_tokens)
            
            neg_texts = [self.negative_prompt.format(name) for name in self.class_names]
            neg_tokens = clip.tokenize(neg_texts).to(self.device)
            neg_features = self.clip.encode_text(neg_tokens)
        
        self.register_buffer('pos_text_features', pos_features.float())
        self.register_buffer('neg_text_features', neg_features.float())
    
    def get_projected_text_features(
        self, 
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_proj = self.text_projector(self.pos_text_features)
        neg_proj = self.text_projector(self.neg_text_features)
        
        text_proj_2k = torch.cat([neg_proj, pos_proj], dim=0)
        
        if normalize:
            text_proj_2k = F.normalize(text_proj_2k, dim=-1)
            pos_proj = F.normalize(pos_proj, dim=-1)
        
        return text_proj_2k, pos_proj

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()  
        return self
    
    def forward(self, images: torch.Tensor, return_features: bool = False):
        B = images.shape[0]
        K = self.num_classes

        with torch.no_grad():
            img_local = self.clip.encode_image_local(images)      

        img_proj_raw = self.image_projector(img_local.float())    
        img_proj = F.normalize(img_proj_raw, dim=-1)              

        text_2k = torch.cat([self.neg_text_features, self.pos_text_features], dim=0)
        text_proj_2k_raw = self.text_projector(text_2k)                             

        neg_raw = text_proj_2k_raw[:K] 
        pos_raw = text_proj_2k_raw[K:] 

        neg = F.normalize(neg_raw, dim=-1)
        pos = F.normalize(pos_raw, dim=-1)
        text_proj_2k = torch.cat([neg, pos], dim=0)  

        sim = torch.matmul(img_proj, text_proj_2k.t()) 
        sim = sim.permute(0, 2, 1).contiguous()        

        sim_neg = sim[:, :K, :]                    
        sim_pos = sim[:, K:, :]                    

        attn_temp = 5.0
        out_scale = float(self.aggregation_scale)

        sim_neg_t = sim_neg * attn_temp
        sim_pos_t = sim_pos * attn_temp

        q_neg = F.softmax(sim_neg_t, dim=-1)     
        q_pos = F.softmax(sim_pos_t, dim=-1)     

        p_neg = (q_neg * sim_neg_t).sum(dim=-1) * out_scale 
        p_pos = (q_pos * sim_pos_t).sum(dim=-1) * out_scale 

        logits = torch.stack([p_neg, p_pos], dim=1)      

        if return_features:
            return logits, text_proj_2k_raw, F.normalize(pos_raw, dim=-1)

        return logits


    
    def predict(
        self, 
        images: torch.Tensor, 
        threshold: float = 0.5
    ) -> torch.Tensor:
        logits = self.forward(images, return_features=False)
        probs = F.softmax(logits, dim=1)[:, 1, :]
        return (probs > threshold).float()


def create_dclip_model(
    clip_model_name: str = "RN101",
    class_names: List[str] = None,
    num_classes: int = 20,
    proj_dim: int = 256,
    text_hidden_dim: int = 384,
    aggregation_scale: float = 5.0,
    device: str = "cuda",
) -> DCLIP:
    model = DCLIP(
        clip_model_name=clip_model_name,
        num_classes=num_classes,
        class_names=class_names,
        proj_dim=proj_dim,
        text_hidden_dim=text_hidden_dim,
        aggregation_scale=aggregation_scale,
        device=device,
    )
    
    return model.to(device)