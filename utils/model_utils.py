"""
Model utility functions.
"""
import os
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn


def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    """
    Initialize network weights.
    
    Args:
        net: Network to initialize
        init_type: Initialization type ('normal', 'xavier', 'kaiming', 'orthogonal')
        init_gain: Scaling factor for normal, xavier and orthogonal
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method {init_type} is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    
    net.apply(init_func)


def set_requires_grad(nets: list, requires_grad: bool = False) -> None:
    """
    Set requires_grad for a list of networks.
    
    Args:
        nets: List of networks
        requires_grad: Whether to require gradients
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False
) -> None:
    """
    Save checkpoint.
    
    Args:
        state: State dictionary to save
        filepath: Path to save checkpoint
        is_best: Whether this is the best model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    
    if is_best:
        best_path = filepath.replace('.pth', '_best.pth')
        torch.save(state, best_path)


def load_checkpoint(
    checkpoint_path: str,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Load checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
    
    Returns:
        Checkpoint dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Use weights_only=False for PyTorch 2.6+ compatibility with custom classes
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def load_pretrained_model(
    model: nn.Module,
    checkpoint_path: str,
    device: str = "cuda",
    strict: bool = False
) -> nn.Module:
    """
    Load pre-trained model weights.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        device: Device to load on
        strict: Whether to strictly match state dict keys
    
    Returns:
        Model with loaded weights
    """
    checkpoint = load_checkpoint(checkpoint_path, device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'G_AB' in checkpoint or 'G_BA' in checkpoint:
        # CycleGAN format - we'll handle this in the trainer
        state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Try to load generator weights
    if isinstance(state_dict, dict):
        # Look for generator keys
        gen_keys = [k for k in state_dict.keys() if 'generator' in k.lower() or 'G_' in k]
        if gen_keys:
            # Extract generator state dict
            model_state = {}
            for key in state_dict.keys():
                if 'generator' in key.lower() or 'G_' in key:
                    new_key = key.replace('G_AB.', '').replace('G_BA.', '').replace('generator.', '')
                    model_state[new_key] = state_dict[key]
            
            if model_state:
                model.load_state_dict(model_state, strict=strict)
                return model
    
    # Try direct load
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        # If direct load fails, try partial load
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    return model

