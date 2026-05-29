"""
Inference script for CycleGAN Van Gogh style transfer.
"""
import os
import sys
import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as transforms

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import Generator
from utils import load_checkpoint, save_image, tensor_to_image


def get_transform(image_size: int = 256) -> transforms.Compose:
    """Get transform for inference."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def load_model(checkpoint_path: str, device: str = "cuda", ngf: int = 64) -> Generator:
    """
    Load trained generator model.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load on
        ngf: Number of generator filters
    
    Returns:
        Loaded generator model
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Initialize generator
    generator = Generator(
        input_nc=3,
        output_nc=3,
        ngf=ngf
    ).to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device)
    
    # Try different checkpoint formats
    if 'G_AB' in checkpoint:
        generator.load_state_dict(checkpoint['G_AB'], strict=False)
    elif 'generator' in checkpoint:
        generator.load_state_dict(checkpoint['generator'], strict=False)
    elif 'state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        # Try direct load
        generator.load_state_dict(checkpoint, strict=False)
    
    generator.eval()
    return generator


def stylize_image(
    image_path: str,
    checkpoint_path: str,
    output_path: str,
    device: str = "cuda",
    image_size: int = 256,
    ngf: int = 64
) -> None:
    """
    Apply Van Gogh style to an image.
    
    Args:
        image_path: Path to input image
        checkpoint_path: Path to model checkpoint
        output_path: Path to save output image
        device: Device to use
        image_size: Image size
        ngf: Number of generator filters
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Loading model from {checkpoint_path}")
    generator = load_model(checkpoint_path, device, ngf)
    
    # Load and preprocess image
    print(f"Loading image from {image_path}")
    image = Image.open(image_path).convert('RGB')
    transform = get_transform(image_size)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate styled image
    print("Generating styled image...")
    with torch.no_grad():
        styled_tensor = generator(image_tensor)
    
    # Save output
    print(f"Saving output to {output_path}")
    save_image(styled_tensor, output_path)
    print("Done!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="CycleGAN Van Gogh Style Transfer Inference")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output_path", type=str, default="./output_styled.png",
                       help="Path to save output image")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to use")
    parser.add_argument("--image_size", type=int, default=256,
                       help="Image size")
    parser.add_argument("--ngf", type=int, default=64,
                       help="Number of generator filters")
    
    args = parser.parse_args()
    
    stylize_image(
        image_path=args.image_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        device=args.device,
        image_size=args.image_size,
        ngf=args.ngf
    )


if __name__ == "__main__":
    main()

