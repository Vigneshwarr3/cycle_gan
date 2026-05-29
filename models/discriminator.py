"""
Discriminator network for CycleGAN.
Based on PatchGAN architecture.
"""
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator network for CycleGAN.
    PatchGAN architecture: 70x70 patches.
    """
    
    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3
    ):
        """
        Initialize Discriminator.
        
        Args:
            input_nc: Number of input channels
            ndf: Number of discriminator filters in first conv layer
            n_layers: Number of layers in discriminator
        """
        super(Discriminator, self).__init__()
        
        sequence = [
            nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                4,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Output layer
        sequence += [nn.Conv2d(ndf * nf_mult, 1, 4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

