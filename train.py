"""
Training script for CycleGAN with fine-tuning support.
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.hyperparameters import Hyperparameters, get_args, args_to_hyperparameters
from models import Generator, Discriminator
from data import get_dataloaders
from utils import (
    init_weights,
    load_checkpoint,
    save_checkpoint,
    set_requires_grad,
    save_image
)


class CycleGANTrainer:
    """Trainer for CycleGAN model."""
    
    def __init__(self, config: Hyperparameters):
        """
        Initialize trainer.
        
        Args:
            config: Hyperparameters configuration
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Initialize models
        self.G_AB = Generator(
            input_nc=config.input_nc,
            output_nc=config.output_nc,
            ngf=config.ngf
        ).to(self.device)
        
        self.G_BA = Generator(
            input_nc=config.input_nc,
            output_nc=config.output_nc,
            ngf=config.ngf
        ).to(self.device)
        
        self.D_A = Discriminator(
            input_nc=config.input_nc,
            ndf=config.ndf
        ).to(self.device)
        
        self.D_B = Discriminator(
            input_nc=config.input_nc,
            ndf=config.ndf
        ).to(self.device)
        
        # Initialize weights
        init_weights(self.G_AB)
        init_weights(self.G_BA)
        init_weights(self.D_A)
        init_weights(self.D_B)
        
        # Load pre-trained model if specified
        self.start_epoch = 0
        if config.pretrained_path:
            self._load_pretrained(config.pretrained_path)
        
        # Loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # Optimizers
        lr = config.fine_tune_lr if config.pretrained_path else config.lr
        
        self.optimizer_G = optim.Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=lr,
            betas=(config.beta1, config.beta2)
        )
        
        self.optimizer_D_A = optim.Adam(
            self.D_A.parameters(),
            lr=lr,
            betas=(config.beta1, config.beta2)
        )
        
        self.optimizer_D_B = optim.Adam(
            self.D_B.parameters(),
            lr=lr,
            betas=(config.beta1, config.beta2)
        )
        
        # Freeze discriminators if fine-tuning
        if config.freeze_discriminators and config.pretrained_path:
            set_requires_grad([self.D_A, self.D_B], False)
        
        # Learning rate schedulers
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - config.n_epochs) / float(config.n_epochs_decay + 1)
            return lr_l
        
        self.scheduler_G = optim.lr_scheduler.LambdaLR(
            self.optimizer_G,
            lr_lambda=lambda_rule
        )
        self.scheduler_D_A = optim.lr_scheduler.LambdaLR(
            self.optimizer_D_A,
            lr_lambda=lambda_rule
        )
        self.scheduler_D_B = optim.lr_scheduler.LambdaLR(
            self.optimizer_D_B,
            lr_lambda=lambda_rule
        )
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Fake label for discriminator
        self.fake_label = 0.0
        self.real_label = 1.0
        
        # Compute discriminator output size once
        with torch.no_grad():
            dummy_input = torch.zeros(1, config.input_nc, config.image_size, config.image_size).to(self.device)
            dummy_output = self.D_A(dummy_input)
            self.label_size = dummy_output.shape[1:]  # (C, H, W) excluding batch dimension
    
    def _load_pretrained(self, checkpoint_path: str):
        """Load pre-trained model weights."""
        print(f"Loading pre-trained model from {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        
        # Try to load generator weights
        if 'G_AB' in checkpoint:
            self.G_AB.load_state_dict(checkpoint['G_AB'], strict=False)
            print("Loaded G_AB weights")
        if 'G_BA' in checkpoint:
            self.G_BA.load_state_dict(checkpoint['G_BA'], strict=False)
            print("Loaded G_BA weights")
        if 'D_A' in checkpoint and not self.config.freeze_discriminators:
            self.D_A.load_state_dict(checkpoint['D_A'], strict=False)
            print("Loaded D_A weights")
        if 'D_B' in checkpoint and not self.config.freeze_discriminators:
            self.D_B.load_state_dict(checkpoint['D_B'], strict=False)
            print("Loaded D_B weights")
        
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {self.start_epoch}")
    
    def train_epoch(self, dataloader, epoch: int):
        """Train for one epoch."""
        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.n_epochs}")
        
        for i, (real_A, real_B) in enumerate(progress_bar):
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)
            
            batch_size = real_A.size(0)
            
            # Create labels matching discriminator output size
            label_real = torch.full((batch_size, *self.label_size), self.real_label, device=self.device)
            label_fake = torch.full((batch_size, *self.label_size), self.fake_label, device=self.device)
            
            # ========== Train Generators ==========
            set_requires_grad([self.D_A, self.D_B], False)
            self.optimizer_G.zero_grad()
            
            # Identity loss
            loss_idt_A = self.criterion_identity(self.G_BA(real_A), real_A)
            loss_idt_B = self.criterion_identity(self.G_AB(real_B), real_B)
            loss_identity = (loss_idt_A + loss_idt_B) / 2
            
            # GAN loss
            fake_B = self.G_AB(real_A)
            pred_fake = self.D_B(fake_B)
            loss_GAN_AB = self.criterion_GAN(pred_fake, label_real)
            
            fake_A = self.G_BA(real_B)
            pred_fake = self.D_A(fake_A)
            loss_GAN_BA = self.criterion_GAN(pred_fake, label_real)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
            
            # Cycle loss
            rec_A = self.G_BA(fake_B)
            loss_cycle_A = self.criterion_cycle(rec_A, real_A)
            
            rec_B = self.G_AB(fake_A)
            loss_cycle_B = self.criterion_cycle(rec_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            
            # Total generator loss
            loss_G = loss_GAN + self.config.lambda_cycle * loss_cycle + self.config.lambda_identity * loss_identity
            
            loss_G.backward()
            self.optimizer_G.step()
            
            # ========== Train Discriminators ==========
            if not self.config.freeze_discriminators:
                # Discriminator A
                set_requires_grad([self.D_A], True)
                self.optimizer_D_A.zero_grad()
                
                pred_real = self.D_A(real_A)
                loss_D_real = self.criterion_GAN(pred_real, label_real)
                
                pred_fake = self.D_A(fake_A.detach())
                loss_D_fake = self.criterion_GAN(pred_fake, label_fake)
                
                loss_D_A = (loss_D_real + loss_D_fake) * 0.5
                loss_D_A.backward()
                self.optimizer_D_A.step()
                
                # Discriminator B
                set_requires_grad([self.D_B], True)
                self.optimizer_D_B.zero_grad()
                
                pred_real = self.D_B(real_B)
                loss_D_real = self.criterion_GAN(pred_real, label_real)
                
                pred_fake = self.D_B(fake_B.detach())
                loss_D_fake = self.criterion_GAN(pred_fake, label_fake)
                
                loss_D_B = (loss_D_real + loss_D_fake) * 0.5
                loss_D_B.backward()
                self.optimizer_D_B.step()
            else:
                loss_D_A = torch.tensor(0.0)
                loss_D_B = torch.tensor(0.0)
            
            # Logging
            if (i + 1) % self.config.print_freq == 0:
                progress_bar.set_postfix({
                    'G': f'{loss_G.item():.4f}',
                    'D_A': f'{loss_D_A.item():.4f}',
                    'D_B': f'{loss_D_B.item():.4f}',
                    'Cycle': f'{loss_cycle.item():.4f}'
                })
                
                global_step = epoch * len(dataloader) + i
                self.writer.add_scalar('Loss/Generator', loss_G.item(), global_step)
                self.writer.add_scalar('Loss/Discriminator_A', loss_D_A.item(), global_step)
                self.writer.add_scalar('Loss/Discriminator_B', loss_D_B.item(), global_step)
                self.writer.add_scalar('Loss/Cycle', loss_cycle.item(), global_step)
                self.writer.add_scalar('Loss/Identity', loss_identity.item(), global_step)
    
    def train(self):
        """Main training loop."""
        # Get dataloader
        dataloader = get_dataloaders(
            photo_path=self.config.photo_path,
            style_path=self.config.style_path,
            batch_size=self.config.batch_size,
            image_size=self.config.image_size,
            n_threads=self.config.n_threads,
            mode="train"
        )
        
        print(f"Starting training from epoch {self.start_epoch}")
        print(f"Total epochs: {self.config.n_epochs}")
        print(f"Device: {self.device}")
        
        for epoch in range(self.start_epoch, self.config.n_epochs):
            self.train_epoch(dataloader, epoch)
            
            # Update learning rates
            self.scheduler_G.step()
            self.scheduler_D_A.step()
            self.scheduler_D_B.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_epoch_freq == 0:
                checkpoint = {
                    'epoch': epoch,
                    'G_AB': self.G_AB.state_dict(),
                    'G_BA': self.G_BA.state_dict(),
                    'D_A': self.D_A.state_dict(),
                    'D_B': self.D_B.state_dict(),
                    'optimizer_G': self.optimizer_G.state_dict(),
                    'optimizer_D_A': self.optimizer_D_A.state_dict(),
                    'optimizer_D_B': self.optimizer_D_B.state_dict(),
                    'config': self.config
                }
                
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f'checkpoint_epoch_{epoch+1}.pth'
                )
                save_checkpoint(checkpoint, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
            
            # Save sample images
            if (epoch + 1) % 5 == 0:
                self._save_sample_images(epoch)
        
        print("Training completed!")
        self.writer.close()
    
    def _save_sample_images(self, epoch: int):
        """Save sample images for visualization."""
        self.G_AB.eval()
        self.G_BA.eval()
        
        # Get a sample batch
        dataloader = get_dataloaders(
            photo_path=self.config.photo_path,
            style_path=self.config.style_path,
            batch_size=1,
            image_size=self.config.image_size,
            n_threads=1,
            mode="test"
        )
        
        with torch.no_grad():
            real_A, real_B = next(iter(dataloader))
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)
            
            fake_B = self.G_AB(real_A)
            fake_A = self.G_BA(real_B)
            rec_A = self.G_BA(fake_B)
            rec_B = self.G_AB(fake_A)
            
            # Save images
            save_image(real_A, os.path.join(self.config.output_dir, f'epoch_{epoch+1}_real_A.png'))
            save_image(fake_B, os.path.join(self.config.output_dir, f'epoch_{epoch+1}_fake_B.png'))
            save_image(real_B, os.path.join(self.config.output_dir, f'epoch_{epoch+1}_real_B.png'))
            save_image(fake_A, os.path.join(self.config.output_dir, f'epoch_{epoch+1}_fake_A.png'))
        
        self.G_AB.train()
        self.G_BA.train()


def main():
    """Main function."""
    args = get_args()
    config = args_to_hyperparameters(args)
    
    trainer = CycleGANTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

