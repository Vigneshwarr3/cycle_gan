# CycleGAN Van Gogh Style Transfer

A modular PyTorch implementation of CycleGAN for fine-tuning pre-trained models to produce Van Gogh styled images.

## Features

- **Modular Architecture**: Clean, organized code structure
- **Fine-tuning Support**: Load and fine-tune pre-trained CycleGAN models
- **Flexible Configuration**: Easy-to-use hyperparameter configuration
- **Training & Inference**: Complete training pipeline with inference script

## Installation

1. Clone or navigate to the project directory:
```bash
cd cyclegan-vangogh
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

1. Download the dataset using the provided script:
```bash
chmod +x download.sh
./download.sh
```

2. The dataset will be extracted to `dataset/vangogh2photo/` with the following structure:
   - **Style images (Van Gogh)**: `dataset/vangogh2photo/trainA/`
   - **Photo images**: `dataset/vangogh2photo/trainB/`
   - **Test sets**: `dataset/vangogh2photo/testA/` and `testB/`

The download script automatically extracts the dataset to the correct location.

## Usage

### Training from Scratch

```bash
python train.py \
    --photo_path ./dataset/vangogh2photo/trainB \
    --style_path ./dataset/vangogh2photo/trainA \
    --batch_size 1 \
    --n_epochs 100 \
    --image_size 256 \
    --checkpoint_dir ./checkpoints \
    --output_dir ./outputs
```

Or use default paths (no need to specify):
```bash
python train.py --batch_size 1 --n_epochs 100
```

### Fine-tuning a Pre-trained Model

```bash
python train.py \
    --pretrained_path /path/to/pretrained/checkpoint.pth \
    --photo_path ./dataset/vangogh2photo/trainB \
    --style_path ./dataset/vangogh2photo/trainA \
    --batch_size 1 \
    --n_epochs 50 \
    --fine_tune_lr 0.0001 \
    --freeze_discriminators \
    --checkpoint_dir ./checkpoints \
    --output_dir ./outputs
```

Or use default paths:
```bash
python train.py \
    --pretrained_path /path/to/pretrained/checkpoint.pth \
    --batch_size 1 \
    --n_epochs 50 \
    --fine_tune_lr 0.0001 \
    --freeze_discriminators
```

### Inference (Style Transfer)

```bash
python inference.py \
    --image_path /path/to/input/image.jpg \
    --checkpoint_path ./checkpoints/checkpoint_epoch_100.pth \
    --output_path ./output_styled.png
```

## Configuration Options

### Training Parameters

- `--photo_path`: Path to photo images directory
- `--style_path`: Path to Van Gogh style images directory
- `--batch_size`: Batch size (default: 1)
- `--n_epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.0002)
- `--lambda_cycle`: Cycle consistency loss weight (default: 10.0)
- `--lambda_identity`: Identity loss weight (default: 0.5)
- `--image_size`: Input image size (default: 256)

### Fine-tuning Parameters

- `--pretrained_path`: Path to pre-trained model checkpoint
- `--fine_tune_lr`: Learning rate for fine-tuning (default: 0.0001)
- `--freeze_discriminators`: Freeze discriminators during fine-tuning
- `--resume_training`: Resume training from checkpoint

### Model Architecture

- `--ngf`: Number of generator filters (default: 64)
- `--ndf`: Number of discriminator filters (default: 64)
- `--input_nc`: Number of input channels (default: 3)
- `--output_nc`: Number of output channels (default: 3)

## Model Architecture

### Generator
- ResNet-based architecture with 9 residual blocks
- Architecture: c7s1-64, d128, d256, R256×9, u128, u64, c7s1-3
- Uses instance normalization and reflection padding

### Discriminator
- PatchGAN architecture (70×70 patches)
- 3-layer discriminator with instance normalization

## Training Process

The training process includes:
1. **Adversarial Loss**: Generator vs Discriminator
2. **Cycle Consistency Loss**: Ensures A→B→A reconstruction
3. **Identity Loss**: Preserves color and tone

Losses are logged to TensorBoard (check `./logs/`).

## Checkpoints

Checkpoints are saved every `save_epoch_freq` epochs (default: 5) in the `checkpoint_dir`. Each checkpoint contains:
- Generator weights (G_AB, G_BA)
- Discriminator weights (D_A, D_B)
- Optimizer states
- Training epoch

## Fine-tuning Tips

1. **Lower Learning Rate**: Use `--fine_tune_lr 0.0001` (lower than training from scratch)
2. **Freeze Discriminators**: Use `--freeze_discriminators` to only fine-tune generators
3. **Fewer Epochs**: Fine-tuning typically requires fewer epochs (20-50)
4. **Pre-trained Models**: You can use CycleGAN pre-trained models from:
   - Official CycleGAN repository
   - Model Zoo repositories
   - Your own trained models

## Output

- **Checkpoints**: Saved in `./checkpoints/`
- **Sample Images**: Saved in `./outputs/` during training
- **TensorBoard Logs**: Saved in `./logs/`

