import os
import glob
from pathlib import Path
from typing import Optional, List, Tuple, Union, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from datasets import load_dataset

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class OrientationDataset(Dataset):
    """
    Dataset for document orientation classification.
    Takes source images (assumed upright), and for each source image,
    it effectively generates 4 samples rotated at 0, 90, 180, and 270 degrees.
    Labels: 0: 0 deg, 1: 90 deg, 2: 180 deg, 3: 270 deg.
    """
    def __init__(
        self,
        source: str = 'hf',
        data_dir: Optional[str] = None,
        num_images: int = 5000,
        is_train: bool = True
    ):
        """
        Args:
            source: Data source, 'hf' for HuggingFace rvl_cdip, or 'local' for local directory.
            data_dir: Local directory path if source='local'.
            num_images: Number of source images to use (resulting in 4 * num_images samples).
            is_train: Whether this dataset is used for training (applies augmentations).
        """
        self.is_train = is_train
        self.images: List[Image.Image] = []
        
        if source == 'hf':
            print(f"Loading {num_images} images from Hugging Face rvl_cdip dataset...")
            dataset = self._load_hf_dataset()
            iterator = iter(dataset)
            for _ in range(num_images):
                try:
                    item = next(iterator)
                    img = item['image']
                    if not isinstance(img, Image.Image):
                        import io
                        img = Image.open(io.BytesIO(img['bytes']))
                    self.images.append(img.copy())
                except StopIteration:
                    break
        elif source == 'local':
            if not data_dir:
                raise ValueError("data_dir must be provided if source is 'local'")
            print(f"Loading up to {num_images} images from local directory {data_dir}...")
            image_paths = []
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'):
                image_paths.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
            
            image_paths = image_paths[:num_images]
            for path in image_paths:
                try:
                    img = Image.open(path).copy()
                    self.images.append(img)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
        else:
            raise ValueError(f"Unknown source: {source}")
            
        print(f"Loaded {len(self.images)} source images. Total dataset size will be {len(self.images) * 4}.")

        # Base transforms applied to all images
        # Resize, convert to tensor, normalize with ImageNet stats
        self.base_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Augmentation applied only during training
        if self.is_train:
            self.aug_transforms = transforms.Compose([
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
            self.noise_transform = AddGaussianNoise(0., 0.05)
        else:
            self.aug_transforms = None
            self.noise_transform = None

    @staticmethod
    def _load_hf_dataset():
        """Load document images from HuggingFace Hub.

        Uses aharley/rvl_cdip (community re-upload in modern Parquet format).
        Falls back to dvgodoy/rvl_cdip_mini (1% subset) if the full dataset fails.
        """
        # Primary: community Parquet re-upload (400k document images)
        try:
            print("  Loading aharley/rvl_cdip (Parquet format)...")
            ds = load_dataset("aharley/rvl_cdip", split="train", streaming=True)
            print("  Dataset loaded successfully.")
            return ds
        except Exception as e:
            print(f"  aharley/rvl_cdip failed: {e}")

        # Fallback: 1% mini subset
        try:
            print("  Trying dvgodoy/rvl_cdip_mini...")
            ds = load_dataset("dvgodoy/rvl_cdip_mini", split="train", streaming=True)
            print("  Mini dataset loaded successfully.")
            return ds
        except Exception as e:
            print(f"  dvgodoy/rvl_cdip_mini failed: {e}")

        raise RuntimeError(
            "Could not load any document dataset from HuggingFace.\n"
            "Please use --data-source local with a directory of document images."
        )

    def __len__(self) -> int:
        return len(self.images) * 4

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_idx = idx // 4
        rotation_idx = idx % 4
        
        img = self.images[img_idx]
        
        # Convert to RGB if grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Rotate image based on rotation_idx
        # Labels: 0=0°, 1=90°, 2=180°, 3=270°
        # Note: PIL.Image.rotate rotates counter-clockwise.
        # We define label 1 as 90 deg rotation. 
        if rotation_idx == 1:
            img = img.rotate(90, expand=True)
        elif rotation_idx == 2:
            img = img.rotate(180, expand=True)
        elif rotation_idx == 3:
            img = img.rotate(270, expand=True)
            
        # Apply transforms
        if self.is_train and self.aug_transforms:
            img = self.aug_transforms(img)
            
        tensor = self.base_transforms(img)
        
        if self.is_train and self.noise_transform:
            # Apply noise with 50% probability
            if torch.rand(1).item() < 0.5:
                tensor = self.noise_transform(tensor)
                
        return tensor, rotation_idx

def create_splits(
    source: str = 'hf',
    data_dir: Optional[str] = None,
    num_images: int = 5000,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Creates train, validation, and test datasets with a 70/15/15 split.
    """
    # Create the full dataset with is_train=True to load the images
    # We will wrap it to properly handle is_train for val/test splits
    full_dataset = OrientationDataset(source=source, data_dir=data_dir, num_images=num_images, is_train=True)
    
    total_len = len(full_dataset.images)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    
    # We need to split the source images, NOT the multiplied dataset
    # So we split indices of the source images
    indices = torch.randperm(total_len, generator=torch.Generator().manual_seed(seed)).tolist()
    
    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len+val_len]
    test_indices = indices[train_len+val_len:]
    
    # Create copies of the dataset for different splits to isolate their images and train flags
    class SubsetOrientationDataset(Dataset):
        def __init__(self, parent_dataset: OrientationDataset, subset_indices: List[int], is_train: bool):
            self.parent = parent_dataset
            self.subset_indices = subset_indices
            self.is_train = is_train
            
        def __len__(self):
            return len(self.subset_indices) * 4
            
        def __getitem__(self, idx: int):
            img_idx = self.subset_indices[idx // 4]
            rotation_idx = idx % 4
            
            img = self.parent.images[img_idx]
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            if rotation_idx == 1:
                img = img.rotate(90, expand=True)
            elif rotation_idx == 2:
                img = img.rotate(180, expand=True)
            elif rotation_idx == 3:
                img = img.rotate(270, expand=True)
                
            if self.is_train and self.parent.aug_transforms:
                img = self.parent.aug_transforms(img)
                
            tensor = self.parent.base_transforms(img)
            
            if self.is_train and self.parent.noise_transform:
                if torch.rand(1).item() < 0.5:
                    tensor = self.parent.noise_transform(tensor)
                    
            return tensor, rotation_idx

    train_dataset = SubsetOrientationDataset(full_dataset, train_indices, is_train=True)
    val_dataset = SubsetOrientationDataset(full_dataset, val_indices, is_train=False)
    test_dataset = SubsetOrientationDataset(full_dataset, test_indices, is_train=False)
    
    return train_dataset, val_dataset, test_dataset
