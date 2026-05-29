"""
Image utility functions.
"""
import os
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy image.
    
    Args:
        tensor: Tensor of shape (C, H, W) or (B, C, H, W)
    
    Returns:
        Numpy array of shape (H, W, C) or (B, H, W, C)
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Denormalize
    image = tensor.cpu().float().numpy()
    image = (image + 1) / 2.0  # [-1, 1] -> [0, 1]
    image = np.clip(image, 0, 1)
    
    # Convert to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    
    return image


def save_image(
    tensor: torch.Tensor,
    path: str,
    nrow: int = 1,
    normalize: bool = True
) -> None:
    """
    Save tensor as image.
    
    Args:
        tensor: Tensor to save
        path: Path to save image
        nrow: Number of images per row
        normalize: Whether to normalize
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    image = tensor_to_image(tensor)
    image = (image * 255).astype(np.uint8)
    
    img = Image.fromarray(image)
    img.save(path)

