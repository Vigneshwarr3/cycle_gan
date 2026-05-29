"""
Generator network for CycleGAN.
Based on ResNet architecture with 9 residual blocks.
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""
    
    def __init__(self, in_channels: int):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    """
    Generator network for CycleGAN.
    Architecture: c7s1-64, d128, d256, R256 x 9, u128, u64, c7s1-3
    """
    
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        n_blocks: int = 9,
        use_dropout: bool = False
    ):
        """
        Initialize Generator.
        
        Args:
            input_nc: Number of input channels
            output_nc: Number of output channels
            ngf: Number of generator filters in first conv layer
            n_blocks: Number of residual blocks
            use_dropout: Whether to use dropout in residual blocks
        """
        super(Generator, self).__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]
        
        # Residual blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResidualBlock(ngf * mult)]
        
        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(inplace=True)
            ]
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

