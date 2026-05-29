"""
Dataset and DataLoader for CycleGAN training.
"""
import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    """Dataset for loading images from two domains."""
    
    def __init__(
        self,
        root_A: str,
        root_B: str,
        transform: Optional[transforms.Compose] = None,
        mode: str = "train"
    ):
        """
        Initialize dataset.
        
        Args:
            root_A: Root directory for domain A images
            root_B: Root directory for domain B images
            transform: Transform to apply to images
            mode: 'train' or 'test'
        """
        self.root_A = Path(root_A)
        self.root_B = Path(root_B)
        self.transform = transform
        self.mode = mode
        
        # Get list of image files
        self.A_paths = sorted(self._get_image_paths(self.root_A))
        self.B_paths = sorted(self._get_image_paths(self.root_B))
        
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
    
    def _get_image_paths(self, root: Path) -> list:
        """Get all image file paths from directory."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        paths = []
        
        if root.exists():
            for ext in extensions:
                paths.extend(root.glob(f'*{ext}'))
                paths.extend(root.glob(f'*{ext.upper()}'))
        
        return [str(p) for p in paths]
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item at index."""
        A_path = self.A_paths[index % self.A_size]
        
        # Random index for domain B
        index_B = torch.randint(0, self.B_size, (1,)).item()
        B_path = self.B_paths[index_B]
        
        # Load images
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)
        
        return A_img, B_img
    
    def __len__(self) -> int:
        """Return dataset size."""
        return max(self.A_size, self.B_size)


def get_transform(
    image_size: int = 256,
    mode: str = "train",
    normalize: bool = True
) -> transforms.Compose:
    """
    Get transform for images.
    
    Args:
        image_size: Target image size
        mode: 'train' or 'test'
        normalize: Whether to normalize images
    """
    transform_list = []
    
    if mode == "train":
        transform_list.extend([
            transforms.Resize(int(image_size * 1.12)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        transform_list.append(transforms.Resize(image_size))
        transform_list.append(transforms.CenterCrop(image_size))
    
    transform_list.extend([
        transforms.ToTensor(),
    ])
    
    if normalize:
        transform_list.append(
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )
    
    return transforms.Compose(transform_list)


def get_dataloaders(
    photo_path: str,
    style_path: str,
    batch_size: int = 1,
    image_size: int = 256,
    n_threads: int = 4,
    mode: str = "train"
) -> Tuple[DataLoader, DataLoader]:
    """
    Get DataLoaders for photo and style images.
    
    Args:
        photo_path: Path to photo images
        style_path: Path to style images
        batch_size: Batch size
        image_size: Image size
        n_threads: Number of data loading threads
        mode: 'train' or 'test'
    
    Returns:
        Tuple of (photo_loader, style_loader)
    """
    transform = get_transform(image_size, mode)
    
    dataset = ImageDataset(
        root_A=photo_path,
        root_B=style_path,
        transform=transform,
        mode=mode
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=n_threads,
        pin_memory=True
    )
    
    return dataloader

