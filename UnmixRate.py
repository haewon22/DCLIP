"""
UnmixRate - Fine-grained Discrimination Metric for VLMs

Purpose:
    기존 mAP, mIoU 등의 metric이 제공하지 못하는 
    "모델이 유사한 객체를 얼마나 잘 구분하는지"를 정량화하는 새로운 평가 지표.

Core Idea:
    1. 타겟(Positive) 클래스에 대해 가장 혼동하기 쉬운 N개의 Negative 클래스를 선정
    2. Zero-shot segmentation으로 각 클래스의 예측 마스크 획득
    3. Positive IoU (타겟을 얼마나 잘 찾았는가)와 
       Negative Confusion (타겟 영역을 얼마나 침범했는가)를 종합 평가

Metrics:
    - Positive IoU: GT 마스크와 타겟 클래스 예측 마스크의 IoU
    - Negative Confusion: GT 마스크와 Negative 클래스들 예측 마스크의 평균 IoU
    - UnmixRate: Positive IoU와 (1 - Negative Confusion)의 조화평균/기하평균
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm

try:
    import clip
except ImportError:
    raise ImportError("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")


# =============================================================================
# Constants
# =============================================================================

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# VOC Segmentation Color Map (21 colors: background + 20 classes)
def get_voc_colormap() -> np.ndarray:
    """PASCAL VOC 21-color colormap 반환. Shape: (21, 3), dtype: uint8"""
    full_map = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        r = g = b = 0
        cid = i
        for j in range(8):
            r |= ((cid >> 0) & 1) << (7 - j)
            g |= ((cid >> 1) & 1) << (7 - j)
            b |= ((cid >> 2) & 1) << (7 - j)
            cid >>= 3
        full_map[i] = [r, g, b]
    return full_map[:21]

VOC_COLORMAP = get_voc_colormap()
VOC_COLORMAP_DICT = {name: VOC_COLORMAP[i+1] for i, name in enumerate(VOC_CLASSES)}
VOC_COLORMAP_DICT['background'] = VOC_COLORMAP[0]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class UnmixRateResult:
    """단일 이미지-클래스 쌍에 대한 UnmixRate 결과"""
    image_id: str
    target_class: str
    negative_classes: List[str]
    positive_iou: float
    negative_ious: Dict[str, float]
    avg_negative_iou: float
    non_confusion_score: float  # 1 - avg_negative_iou
    unmix_harmonic: float       # 조화평균
    unmix_geometric: float      # 기하평균
    
    def to_dict(self) -> dict:
        return {
            'image_id': self.image_id,
            'target_class': self.target_class,
            'negative_classes': ','.join(self.negative_classes),
            'positive_iou': self.positive_iou,
            'avg_negative_iou': self.avg_negative_iou,
            'non_confusion_score': self.non_confusion_score,
            'unmix_harmonic': self.unmix_harmonic,
            'unmix_geometric': self.unmix_geometric,
        }


@dataclass 
class UnmixRateSummary:
    """전체 데이터셋에 대한 UnmixRate 요약 통계"""
    num_samples: int
    avg_positive_iou: float
    avg_negative_iou: float
    avg_non_confusion: float
    avg_unmix_harmonic: float
    avg_unmix_geometric: float
    per_class_unmixrate: Dict[str, float]  # 클래스별 UnmixRate


# =============================================================================
# Similarity Database
# =============================================================================

class SimilarityDatabase:
    """
    클래스 간 의미론적 유사도를 관리하는 DB.
    CLIP text encoder를 사용해 클래스 간 cosine similarity 계산.
    """
    
    def __init__(
        self, 
        clip_model: torch.nn.Module,
        class_names: List[str] = VOC_CLASSES,
        prompt_template: str = "A photo of a {}.",
        device: str = "cuda"
    ):
        self.class_names = class_names
        self.prompt_template = prompt_template
        self.device = device
        self.clip_model = clip_model
        self.similarity_matrix: Optional[np.ndarray] = None
        self._compute_similarity()
    
    def _compute_similarity(self):
        """CLIP text encoder로 클래스 간 유사도 행렬 계산"""
        texts = [self.prompt_template.format(name) for name in self.class_names]
        tokens = clip.tokenize(texts).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
            text_features = F.normalize(text_features.float(), dim=-1)
            self.similarity_matrix = (text_features @ text_features.T).cpu().numpy()
    
    def get_most_similar_negatives(
        self, 
        target_class: str, 
        n: int = 3
    ) -> List[str]:
        """
        타겟 클래스와 가장 유사한 N개의 다른 클래스 반환.
        모델이 가장 혼동하기 쉬운 클래스들.
        """
        if target_class not in self.class_names:
            raise ValueError(f"Unknown class: {target_class}")
        
        target_idx = self.class_names.index(target_class)
        sims = self.similarity_matrix[target_idx].copy()
        sims[target_idx] = -1  # 자기 자신 제외
        
        # 유사도 높은 순으로 정렬
        sorted_indices = np.argsort(sims)[::-1]
        
        return [self.class_names[i] for i in sorted_indices[:n]]
    
    def get_similarity(self, class_a: str, class_b: str) -> float:
        """두 클래스 간 유사도 반환"""
        idx_a = self.class_names.index(class_a)
        idx_b = self.class_names.index(class_b)
        return float(self.similarity_matrix[idx_a, idx_b])
    
    def save_to_csv(self, output_path: str):
        """유사도 행렬을 CSV로 저장"""
        records = []
        for i, cls_i in enumerate(self.class_names):
            for j, cls_j in enumerate(self.class_names):
                records.append({
                    'class': cls_i,
                    'other_class': cls_j,
                    'cosine_similarity': float(self.similarity_matrix[i, j])
                })
        
        df = pd.DataFrame(records)
        df_sorted = df.sort_values(['class', 'cosine_similarity'], ascending=[True, False])
        df_sorted.to_csv(output_path, index=False)
        print(f"Similarity matrix saved to {output_path}")
    
    @classmethod
    def load_from_csv(cls, csv_path: str, class_names: List[str] = VOC_CLASSES):
        """CSV에서 유사도 행렬 로드"""
        df = pd.read_csv(csv_path)
        instance = object.__new__(cls)
        instance.class_names = class_names
        instance.similarity_matrix = np.zeros((len(class_names), len(class_names)))
        
        for _, row in df.iterrows():
            if row['class'] in class_names and row['other_class'] in class_names:
                i = class_names.index(row['class'])
                j = class_names.index(row['other_class'])
                instance.similarity_matrix[i, j] = row['cosine_similarity']
        
        return instance


# =============================================================================
# Zero-Shot Segmentation
# =============================================================================

class ZeroShotSegmenter:
    """
    CLIP 기반 Zero-Shot Semantic Segmentation.
    이미지의 local features와 텍스트 features 간 유사도로 마스크 생성.
    """
    
    def __init__(
        self,
        clip_model: torch.nn.Module,
        preprocess: callable,
        device: str = "cuda",
        use_surgery: bool = False  # CLIP Surgery 사용 여부
    ):
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.device = device
        self.use_surgery = use_surgery
        self.clip_model.eval()
    
    def _get_image_local_features(
        self, 
        image: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        이미지의 local (patch-wise) features 추출.
        
        Returns:
            features: (1, H*W, D) 형태의 local features
            spatial_size: (H, W) 공간 크기
        """
        visual = self.clip_model.visual
        
        # ResNet 기반
        if hasattr(visual, 'layer4'):
            x = image.type(visual.conv1.weight.dtype)
            
            # Stem
            for conv, bn in [(visual.conv1, visual.bn1), 
                            (visual.conv2, visual.bn2), 
                            (visual.conv3, visual.bn3)]:
                x = visual.relu(bn(conv(x)))
            x = visual.avgpool(x)
            
            # ResNet layers
            x = visual.layer1(x)
            x = visual.layer2(x)
            x = visual.layer3(x)
            x = visual.layer4(x)
            
            B, C, H, W = x.shape
            
            # Attention pooling의 value projection만 사용
            attnpool = visual.attnpool
            x = x.reshape(B, C, H * W).permute(2, 0, 1)
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
            
            # Positional embedding 보간
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
            
            # V projection만 사용 (attention 없이)
            x = F.linear(x, attnpool.v_proj.weight, attnpool.v_proj.bias)
            x = F.linear(x, attnpool.c_proj.weight, attnpool.c_proj.bias)
            
            x = x[1:]  # CLS 토큰 제외
            x = x.permute(1, 0, 2)  # (B, H*W, D)
            
            return x, (H, W)
        
        # ViT 기반
        else:
            x = image.type(visual.conv1.weight.dtype)
            x = visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            
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
            
            x = x[:, 1:, :]  # CLS 제외
            if visual.proj is not None:
                x = x @ visual.proj
            
            H = W = int(x.shape[1] ** 0.5)
            return x, (H, W)
    
    def _get_text_features(self, class_names: List[str]) -> torch.Tensor:
        """텍스트 features 추출"""
        texts = [f"A photo of a {name}." for name in class_names]
        tokens = clip.tokenize(texts).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
            text_features = F.normalize(text_features.float(), dim=-1)
        
        return text_features
    
    def segment(
        self,
        image: Union[Image.Image, torch.Tensor],
        class_names: List[str],
        threshold: float = 0.5,
        return_probs: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Zero-shot segmentation 수행.
        
        Args:
            image: PIL Image 또는 전처리된 텐서
            class_names: 분할할 클래스 이름들
            threshold: 마스크 이진화 임계값
            return_probs: True면 확률맵 반환, False면 이진 마스크 반환
        
        Returns:
            {class_name: mask} 딕셔너리
        """
        if isinstance(image, Image.Image):
            original_size = image.size[::-1]  # (H, W)
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        else:
            image_tensor = image.to(self.device)
            original_size = None
        
        with torch.no_grad():
            # Local features 추출
            img_features, (H, W) = self._get_image_local_features(image_tensor)
            img_features = F.normalize(img_features.float(), dim=-1)
            
            # Text features 추출
            text_features = self._get_text_features(class_names)
            
            # Similarity 계산: (B, H*W, num_classes)
            similarity = img_features @ text_features.T
            
            # Reshape to spatial: (B, H, W, num_classes)
            similarity_map = similarity.reshape(1, H, W, len(class_names))
            
            # 원본 크기로 보간
            if original_size is not None:
                similarity_map = similarity_map.permute(0, 3, 1, 2)  # (B, C, H, W)
                similarity_map = F.interpolate(
                    similarity_map, 
                    size=original_size, 
                    mode='bilinear', 
                    align_corners=False
                )
                similarity_map = similarity_map.permute(0, 2, 3, 1)  # (B, H, W, C)
        
        masks = {}
        similarity_map = similarity_map[0].cpu().numpy()  # (H, W, num_classes)
        
        for i, cls_name in enumerate(class_names):
            sim = similarity_map[:, :, i]
            
            # Min-max normalization
            sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
            
            if return_probs:
                masks[cls_name] = sim
            else:
                masks[cls_name] = (sim > threshold).astype(np.uint8) * 255
        
        return masks


# =============================================================================
# UnmixRate Calculator
# =============================================================================

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """두 마스크 간 IoU 계산"""
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_unmixrate_scores(
    positive_iou: float,
    avg_negative_iou: float,
    eps: float = 1e-8
) -> Tuple[float, float]:
    """
    UnmixRate 점수 계산.
    
    Args:
        positive_iou: 타겟 클래스 IoU
        avg_negative_iou: Negative 클래스들의 평균 IoU
    
    Returns:
        (harmonic_mean, geometric_mean)
    """
    non_confusion = 1.0 - avg_negative_iou
    
    # 조화평균
    if positive_iou > eps and non_confusion > eps:
        harmonic = 2.0 / (1.0 / positive_iou + 1.0 / non_confusion)
    else:
        harmonic = 0.0
    
    # 기하평균
    geometric = np.sqrt(positive_iou * non_confusion)
    
    return harmonic, geometric


class UnmixRateEvaluator:
    """
    UnmixRate 평가기.
    
    핵심 프로세스:
    1. 타겟 클래스에 대해 유사한 N개의 Negative 클래스 선정
    2. Zero-shot segmentation으로 각 클래스 마스크 획득
    3. Positive IoU와 Negative Confusion 계산
    4. 조화평균/기하평균으로 최종 UnmixRate 점수 산출
    """
    
    def __init__(
        self,
        clip_model: torch.nn.Module,
        preprocess: callable,
        similarity_db: Optional[SimilarityDatabase] = None,
        class_names: List[str] = VOC_CLASSES,
        n_negative_classes: int = 3,
        threshold: float = 0.5,
        device: str = "cuda"
    ):
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.class_names = class_names
        self.n_negative = n_negative_classes
        self.threshold = threshold
        self.device = device
        
        # Similarity DB 초기화
        if similarity_db is None:
            self.similarity_db = SimilarityDatabase(
                clip_model, class_names, device=device
            )
        else:
            self.similarity_db = similarity_db
        
        # Segmenter 초기화
        self.segmenter = ZeroShotSegmenter(
            clip_model, preprocess, device
        )
    
    def evaluate_single(
        self,
        image: Image.Image,
        gt_mask: np.ndarray,
        target_class: str,
        image_id: str = "unknown"
    ) -> UnmixRateResult:
        """
        단일 이미지-클래스 쌍에 대한 UnmixRate 평가.
        
        Args:
            image: PIL 이미지
            gt_mask: Ground truth 마스크 (H, W), 타겟 클래스 영역이 255
            target_class: 타겟 클래스 이름
            image_id: 이미지 식별자
        
        Returns:
            UnmixRateResult 객체
        """
        # 혼동 유발 Negative 클래스 선정
        negative_classes = self.similarity_db.get_most_similar_negatives(
            target_class, self.n_negative
        )
        
        # 모든 클래스에 대해 segmentation 수행
        all_classes = [target_class] + negative_classes
        masks = self.segmenter.segment(image, all_classes, self.threshold)
        
        # Positive IoU
        positive_mask = masks[target_class]
        positive_iou = compute_iou(gt_mask, positive_mask)
        
        # Negative IoUs
        negative_ious = {}
        for neg_cls in negative_classes:
            neg_mask = masks[neg_cls]
            negative_ious[neg_cls] = compute_iou(gt_mask, neg_mask)
        
        avg_negative_iou = np.mean(list(negative_ious.values()))
        non_confusion = 1.0 - avg_negative_iou
        
        # UnmixRate 점수 계산
        unmix_harmonic, unmix_geometric = compute_unmixrate_scores(positive_iou, avg_negative_iou)
        
        return UnmixRateResult(
            image_id=image_id,
            target_class=target_class,
            negative_classes=negative_classes,
            positive_iou=positive_iou,
            negative_ious=negative_ious,
            avg_negative_iou=avg_negative_iou,
            non_confusion_score=non_confusion,
            unmix_harmonic=unmix_harmonic,
            unmix_geometric=unmix_geometric
        )
    
    def evaluate_dataset(
        self,
        dataset,  # VOCSegmentation 또는 호환 데이터셋
        output_csv: Optional[str] = None,
        verbose: bool = True
    ) -> UnmixRateSummary:
        """
        전체 데이터셋에 대한 UnmixRate 평가.
        
        Args:
            dataset: (image, gt_mask) 반환하는 데이터셋
            output_csv: 결과 저장 경로
            verbose: 진행 상황 출력 여부
        
        Returns:
            UnmixRateSummary 객체
        """
        all_results = []
        
        iterator = tqdm(range(len(dataset)), desc="UnmixRate Evaluation") if verbose else range(len(dataset))
        
        for idx in iterator:
            image_pil, gt_mask_pil = dataset[idx]
            gt_mask_rgb = np.array(gt_mask_pil.convert('RGB'))
            
            # 이미지 ID 추출
            if hasattr(dataset, 'images'):
                image_id = os.path.basename(dataset.images[idx]).split('.')[0]
            else:
                image_id = f"img_{idx:05d}"
            
            # 이미지에 존재하는 클래스 찾기
            present_classes = self._find_present_classes(gt_mask_rgb)
            
            for target_class in present_classes:
                # 해당 클래스의 GT 마스크 추출
                target_mask = self._extract_class_mask(gt_mask_rgb, target_class)
                
                if target_mask.sum() == 0:
                    continue
                
                result = self.evaluate_single(
                    image_pil, target_mask, target_class, image_id
                )
                all_results.append(result)
                
                if verbose and len(all_results) % 50 == 0:
                    print(f"  {target_class}: +IoU={result.positive_iou:.3f}, "
                          f"-IoU={result.avg_negative_iou:.3f}, "
                          f"UnmixRate={result.unmix_harmonic:.3f}")
        
        # Summary 계산
        summary = self._compute_summary(all_results)
        
        # CSV 저장
        if output_csv:
            df = pd.DataFrame([r.to_dict() for r in all_results])
            df.to_csv(output_csv, index=False)
            print(f"\nResults saved to {output_csv}")
        
        return summary
    
    def _find_present_classes(self, gt_mask_rgb: np.ndarray) -> List[str]:
        """GT 마스크에서 존재하는 클래스들 찾기"""
        present = []
        unique_colors = np.unique(gt_mask_rgb.reshape(-1, 3), axis=0)
        
        for color in unique_colors:
            for cls_name, cls_color in VOC_COLORMAP_DICT.items():
                if cls_name == 'background':
                    continue
                if np.array_equal(color, cls_color):
                    present.append(cls_name)
                    break
        
        return present
    
    def _extract_class_mask(
        self, 
        gt_mask_rgb: np.ndarray, 
        class_name: str
    ) -> np.ndarray:
        """특정 클래스의 마스크 추출"""
        target_color = VOC_COLORMAP_DICT[class_name]
        mask = np.all(gt_mask_rgb == target_color, axis=-1)
        return (mask * 255).astype(np.uint8)
    
    def _compute_summary(self, results: List[UnmixRateResult]) -> UnmixRateSummary:
        """전체 결과에서 요약 통계 계산"""
        if not results:
            return UnmixRateSummary(0, 0, 0, 0, 0, 0, {})
        
        pos_ious = [r.positive_iou for r in results]
        neg_ious = [r.avg_negative_iou for r in results]
        non_conf = [r.non_confusion_score for r in results]
        harmonics = [r.unmix_harmonic for r in results]
        geometrics = [r.unmix_geometric for r in results]
        
        # 클래스별 UnmixRate
        per_class = {}
        for cls_name in self.class_names:
            cls_results = [r for r in results if r.target_class == cls_name]
            if cls_results:
                per_class[cls_name] = np.mean([r.unmix_harmonic for r in cls_results])
        
        return UnmixRateSummary(
            num_samples=len(results),
            avg_positive_iou=np.mean(pos_ious),
            avg_negative_iou=np.mean(neg_ious),
            avg_non_confusion=np.mean(non_conf),
            avg_unmix_harmonic=np.mean(harmonics),
            avg_unmix_geometric=np.mean(geometrics),
            per_class_unmixrate=per_class
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_unmixrate_evaluator(
    model_name: str = "RN101",
    n_negative_classes: int = 3,
    threshold: float = 0.5,
    device: str = "cuda"
) -> UnmixRateEvaluator:
    """
    UnmixRate 평가기 간편 생성.
    
    Args:
        model_name: CLIP 모델 이름 (RN101, ViT-B/16, etc.)
        n_negative_classes: 혼동 유발 클래스 수
        threshold: 마스크 이진화 임계값
        device: 디바이스
    
    Returns:
        UnmixRateEvaluator 인스턴스
    """
    clip_model, preprocess = clip.load(model_name, device=device)
    
    return UnmixRateEvaluator(
        clip_model=clip_model,
        preprocess=preprocess,
        n_negative_classes=n_negative_classes,
        threshold=threshold,
        device=device
    )


def run_unmixrate_evaluation(
    voc_root: str = "./data/VOCdevkit",
    year: str = "2007",
    image_set: str = "val",
    model_name: str = "RN101",
    n_negative: int = 3,
    threshold: float = 0.5,
    output_dir: str = "./unmixrate_results",
    device: str = "cuda"
) -> UnmixRateSummary:
    """
    VOC 데이터셋에 대한 UnmixRate 평가 실행.
    
    Args:
        voc_root: VOC 데이터 루트 경로
        year: VOC 연도 (2007 or 2012)
        image_set: 데이터 분할 (train, val, test)
        model_name: CLIP 모델 이름
        n_negative: 혼동 유발 클래스 수
        threshold: 마스크 이진화 임계값
        output_dir: 결과 저장 디렉토리
        device: 디바이스
    
    Returns:
        UnmixRateSummary 객체
    """
    try:
        from torchvision.datasets import VOCSegmentation
    except ImportError:
        raise ImportError("torchvision is required for VOC dataset")
    
    print("=" * 60)
    print("UnmixRate Evaluation")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: VOC{year} {image_set}")
    print(f"N negative classes: {n_negative}")
    print(f"Threshold: {threshold}")
    print("=" * 60)
    
    # 데이터셋 로드
    dataset = VOCSegmentation(
        root=voc_root,
        year=year,
        image_set=image_set,
        download=False
    )
    print(f"Loaded {len(dataset)} images")
    
    # 평가기 생성
    evaluator = create_unmixrate_evaluator(
        model_name=model_name,
        n_negative_classes=n_negative,
        threshold=threshold,
        device=device
    )
    
    # 결과 저장 경로
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(
        output_dir, 
        f"unmixrate_voc{year}_{image_set}_{model_name.replace('/', '-')}_n{n_negative}_t{threshold}.csv"
    )
    
    # 평가 실행
    summary = evaluator.evaluate_dataset(dataset, output_csv=output_csv)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("UnmixRate Evaluation Results")
    print("=" * 60)
    print(f"Total samples: {summary.num_samples}")
    print(f"Average Positive IoU: {summary.avg_positive_iou:.4f}")
    print(f"Average Negative IoU: {summary.avg_negative_iou:.4f}")
    print(f"Average Non-Confusion: {summary.avg_non_confusion:.4f}")
    print(f"Average UnmixRate (Harmonic): {summary.avg_unmix_harmonic:.4f}")
    print(f"Average UnmixRate (Geometric): {summary.avg_unmix_geometric:.4f}")
    print("\nPer-class UnmixRate (Harmonic):")
    for cls_name, unmixrate in sorted(summary.per_class_unmixrate.items(), key=lambda x: -x[1]):
        print(f"  {cls_name:15s}: {unmixrate:.4f}")
    print("=" * 60)
    
    return summary


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="UnmixRate Evaluation")
    parser.add_argument("--voc_root", type=str, default="./data/VOCdevkit")
    parser.add_argument("--year", type=str, default="2007")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--model", type=str, default="RN101")
    parser.add_argument("--n_negative", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="./unmixrate_results")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    run_unmixrate_evaluation(
        voc_root=args.voc_root,
        year=args.year,
        image_set=args.split,
        model_name=args.model,
        n_negative=args.n_negative,
        threshold=args.threshold,
        output_dir=args.output_dir,
        device=args.device
    )
