"""
Hyperparameters configuration for CycleGAN training and fine-tuning.
"""
import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class Hyperparameters:
    """Configuration class for CycleGAN hyperparameters."""
    
    # Dataset paths
    dataset_path: str = "./dataset/vangogh2photo"
    style_path: str = "./dataset/vangogh2photo/trainA"
    photo_path: str = "./dataset/vangogh2photo/trainB"
    
    # Model architecture
    input_nc: int = 3  # Number of input channels
    output_nc: int = 3  # Number of output channels
    ngf: int = 64  # Number of generator filters in first conv layer
    ndf: int = 64  # Number of discriminator filters in first conv layer
    
    # Training parameters
    batch_size: int = 1
    n_epochs: int = 100
    n_epochs_decay: int = 100  # Number of epochs to decay learning rate
    lr: float = 0.0002  # Initial learning rate
    beta1: float = 0.5  # Beta1 for Adam optimizer
    beta2: float = 0.999  # Beta2 for Adam optimizer
    
    # Loss weights
    lambda_cycle: float = 10.0  # Weight for cycle consistency loss
    lambda_identity: float = 0.5  # Weight for identity loss
    
    # Pre-trained model
    pretrained_path: Optional[str] = None  # Path to pre-trained model checkpoint
    resume_training: bool = False  # Whether to resume from checkpoint
    
    # Training settings
    n_threads: int = 4  # Number of data loading threads
    image_size: int = 256  # Input image size
    save_epoch_freq: int = 5  # Frequency of saving checkpoints
    print_freq: int = 100  # Frequency of printing losses
    
    # Device
    device: str = "cuda"  # cuda or cpu
    
    # Output paths
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    
    # Fine-tuning specific
    freeze_discriminators: bool = False  # Freeze discriminators during fine-tuning
    fine_tune_lr: float = 0.0001  # Lower learning rate for fine-tuning


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CycleGAN Van Gogh Style Transfer")
    
    # Dataset
    parser.add_argument("--dataset_path", type=str, default="./dataset/vangogh2photo",
                       help="Path to dataset directory")
    parser.add_argument("--style_path", type=str, default="./dataset/vangogh2photo/trainA",
                       help="Path to style images (Van Gogh)")
    parser.add_argument("--photo_path", type=str, default="./dataset/vangogh2photo/trainB",
                       help="Path to photo images")
    
    # Model
    parser.add_argument("--input_nc", type=int, default=3,
                       help="Number of input channels")
    parser.add_argument("--output_nc", type=int, default=3,
                       help="Number of output channels")
    parser.add_argument("--ngf", type=int, default=64,
                       help="Number of generator filters")
    parser.add_argument("--ndf", type=int, default=64,
                       help="Number of discriminator filters")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--n_epochs_decay", type=int, default=100,
                       help="Number of epochs to decay learning rate")
    parser.add_argument("--lr", type=float, default=0.0002,
                       help="Initial learning rate")
    parser.add_argument("--beta1", type=float, default=0.5,
                       help="Beta1 for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999,
                       help="Beta2 for Adam optimizer")
    
    # Loss weights
    parser.add_argument("--lambda_cycle", type=float, default=10.0,
                       help="Weight for cycle consistency loss")
    parser.add_argument("--lambda_identity", type=float, default=0.5,
                       help="Weight for identity loss")
    
    # Pre-trained model
    parser.add_argument("--pretrained_path", type=str, default=None,
                       help="Path to pre-trained model checkpoint")
    parser.add_argument("--resume_training", action="store_true",
                       help="Resume training from checkpoint")
    
    # Settings
    parser.add_argument("--n_threads", type=int, default=4,
                       help="Number of data loading threads")
    parser.add_argument("--image_size", type=int, default=256,
                       help="Input image size")
    parser.add_argument("--save_epoch_freq", type=int, default=5,
                       help="Frequency of saving checkpoints")
    parser.add_argument("--print_freq", type=int, default=100,
                       help="Frequency of printing losses")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to use (cuda or cpu)")
    
    # Output paths
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Directory to save outputs")
    parser.add_argument("--log_dir", type=str, default="./logs",
                       help="Directory to save logs")
    
    # Fine-tuning
    parser.add_argument("--freeze_discriminators", action="store_true",
                       help="Freeze discriminators during fine-tuning")
    parser.add_argument("--fine_tune_lr", type=float, default=0.0001,
                       help="Learning rate for fine-tuning")
    
    return parser.parse_args()


def args_to_hyperparameters(args: argparse.Namespace) -> Hyperparameters:
    """Convert argparse Namespace to Hyperparameters dataclass."""
    return Hyperparameters(
        dataset_path=args.dataset_path,
        style_path=args.style_path,
        photo_path=args.photo_path,
        input_nc=args.input_nc,
        output_nc=args.output_nc,
        ngf=args.ngf,
        ndf=args.ndf,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        n_epochs_decay=args.n_epochs_decay,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity,
        pretrained_path=args.pretrained_path,
        resume_training=args.resume_training,
        n_threads=args.n_threads,
        image_size=args.image_size,
        save_epoch_freq=args.save_epoch_freq,
        print_freq=args.print_freq,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        freeze_discriminators=args.freeze_discriminators,
        fine_tune_lr=args.fine_tune_lr,
    )

