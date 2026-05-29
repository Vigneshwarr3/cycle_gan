#!/bin/bash
#SBATCH --job-name=cyclegan_vangogh
#SBATCH --account=c01949
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load modules if needed (adjust based on your cluster)
# module load cuda/11.8
# module load python/3.10

# Activate conda environment
# If using conda environment, uncomment and adjust:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate style_transfer

# Or if using base environment with installed packages:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate base

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Navigate to project directory
cd /N/u/vravirao/BigRed200/style_transfer/cyclegan-vangogh

# Create logs directory if it doesn't exist
mkdir -p logs

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Print Python and PyTorch versions
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not available')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Not available')"

# Training parameters - adjust as needed
# For training from scratch:
python train.py \
    --photo_path ./dataset/vangogh2photo/trainB \
    --style_path ./dataset/vangogh2photo/trainA \
    --batch_size 1 \
    --n_epochs 100 \
    --n_epochs_decay 100 \
    --lr 0.0002 \
    --lambda_cycle 10.0 \
    --lambda_identity 0.5 \
    --image_size 256 \
    --n_threads 4 \
    --device cuda \
    --checkpoint_dir ./checkpoints \
    --output_dir ./outputs \
    --log_dir ./logs \
    --save_epoch_freq 5 \
    --print_freq 100

# For fine-tuning a pre-trained model, uncomment and modify:
# python train.py \
#     --pretrained_path /path/to/pretrained/checkpoint.pth \
#     --photo_path ./dataset/vangogh2photo/trainB \
#     --style_path ./dataset/vangogh2photo/trainA \
#     --batch_size 1 \
#     --n_epochs 50 \
#     --fine_tune_lr 0.0001 \
#     --freeze_discriminators \
#     --device cuda \
#     --checkpoint_dir ./checkpoints \
#     --output_dir ./outputs \
#     --log_dir ./logs

echo "End Time: $(date)"
echo "Job completed!"

