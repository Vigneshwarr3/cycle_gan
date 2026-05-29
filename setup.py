"""
Setup script to verify installation and create necessary directories.
"""
import os
from pathlib import Path


def create_directories():
    """Create necessary directories for training."""
    dirs = [
        "checkpoints",
        "outputs",
        "logs"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created/verified directory: {dir_path}")


def verify_imports():
    """Verify that all modules can be imported."""
    try:
        from config.hyperparameters import Hyperparameters, get_args
        print("✓ Config module imported successfully")
        
        from models import Generator, Discriminator
        print("✓ Models module imported successfully")
        
        from data import ImageDataset, get_dataloaders
        print("✓ Data module imported successfully")
        
        from utils import (
            save_image,
            tensor_to_image,
            init_weights,
            load_checkpoint,
            save_checkpoint,
            set_requires_grad
        )
        print("✓ Utils module imported successfully")
        
        print("\nAll imports successful! ✓")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


if __name__ == "__main__":
    print("Setting up CycleGAN Van Gogh project...")
    print("=" * 50)
    
    print("\n1. Creating directories...")
    create_directories()
    
    print("\n2. Verifying imports...")
    success = verify_imports()
    
    if success:
        print("\n" + "=" * 50)
        print("Setup completed successfully!")
        print("\nNext steps:")
        print("1. Download dataset: ./download.sh")
        print("2. Train model: python train.py")
        print("3. Run inference: python inference.py --image_path <path> --checkpoint_path <path>")
    else:
        print("\n" + "=" * 50)
        print("Setup failed. Please check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

