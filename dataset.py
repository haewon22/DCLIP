import os
from typing import List, Tuple, Optional, Callable, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


class Cutout:
    def __init__(self, n_holes: int = 1, length: int = 112):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[1], img.shape[2]
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img * mask


class VOC2007Dataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "trainval",
        image_size: int = 448,
        transform: Optional[Callable] = None,
        use_cutout: bool = True,
        cutout_n_holes: int = 1,
        cutout_length: int = 112,
        use_randaugment: bool = True,
        randaugment_n: int = 2,
        randaugment_m: int = 9,
    ):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.is_train = split == "trainval"

        self.image_ids = self._load_image_ids()
        self.labels, self.masks = self._load_labels_and_masks()

        self.transform = transform if transform is not None else self._build_transform(
            use_cutout, cutout_n_holes, cutout_length,
            use_randaugment, randaugment_n, randaugment_m
        )

        valid_ratio = float(np.mean([m.mean() for m in self.masks.values()]))
        print(f"Loaded VOC2007 {split}: {len(self.image_ids)} images | "
              f"avg valid-label ratio: {valid_ratio:.3f}")

    def _load_image_ids(self) -> List[str]:
        split_file = os.path.join(self.root, "ImageSets", "Main", f"{self.split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def _load_labels_and_masks(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        labels = {img_id: np.zeros(len(VOC_CLASSES), dtype=np.float32) for img_id in self.image_ids}
        masks = {img_id: np.zeros(len(VOC_CLASSES), dtype=np.float32) for img_id in self.image_ids}

        main_dir = os.path.join(self.root, "ImageSets", "Main")
        id_set = set(self.image_ids)

        for cidx, cname in enumerate(VOC_CLASSES):
            path = os.path.join(main_dir, f"{cname}_{self.split}.txt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing VOC class file: {path}")

            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    img_id = parts[0]
                    if img_id not in id_set:
                        continue

                    lab = int(float(parts[1])) 

                    if lab == 1:
                        labels[img_id][cidx] = 1.0
                        masks[img_id][cidx] = 1.0
                    elif lab == -1:
                        labels[img_id][cidx] = 0.0
                        masks[img_id][cidx] = 1.0
                    else:  
                        labels[img_id][cidx] = 0.0
                        masks[img_id][cidx] = 0.0

        return labels, masks

    def _build_transform(
        self,
        use_cutout: bool,
        cutout_n_holes: int,
        cutout_length: int,
        use_randaugment: bool,
        randaugment_n: int,
        randaugment_m: int,
    ) -> Callable:
        normalize = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        if self.is_train:
            transforms_list = [
                T.Resize((self.image_size, self.image_size)),
                T.RandomHorizontalFlip(),
            ]
            if use_randaugment:
                transforms_list.append(
                    T.RandAugment(num_ops=randaugment_n, magnitude=randaugment_m)
                )
            transforms_list.extend([T.ToTensor(), normalize])
            if use_cutout:
                transforms_list.append(Cutout(cutout_n_holes, cutout_length))
            return T.Compose(transforms_list)

        return T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            normalize,
        ])

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.root, "JPEGImages", f"{img_id}.jpg")
        
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = torch.from_numpy(self.labels[img_id])
        mask = torch.from_numpy(self.masks[img_id])

        return img, label, mask


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    image_size: int = 448,
    num_workers: int = 4,
    use_cutout: bool = True,
    use_randaugment: bool = True,
    cutout_n_holes: int = 1,
    cutout_length: int = 112,
    randaugment_n: int = 2,
    randaugment_m: int = 9,
):
    train_dataset = VOC2007Dataset(
        root=data_root,
        split="trainval",
        image_size=image_size,
        use_cutout=use_cutout,
        use_randaugment=use_randaugment,
        cutout_n_holes=cutout_n_holes,
        cutout_length=cutout_length,
        randaugment_n=randaugment_n,
        randaugment_m=randaugment_m,
    )
    test_dataset = VOC2007Dataset(
        root=data_root,
        split="test",
        image_size=image_size,
        use_cutout=False,
        use_randaugment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, test_loader


def get_voc_dataloaders(cfg):
    return create_dataloaders(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        num_workers=cfg.num_workers,
        use_cutout=cfg.use_cutout,
        use_randaugment=cfg.use_randaugment,
        cutout_n_holes=cfg.cutout_n_holes,
        cutout_length=cfg.cutout_length,
        randaugment_n=cfg.randaugment_n,
        randaugment_m=cfg.randaugment_m,
    )