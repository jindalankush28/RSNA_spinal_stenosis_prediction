#!/bin/bash

#SBATCH --job-name=spinal_models    # Job name
#SBATCH --output=logs/spinal_cov.log # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=24         # Number of CPU cores per task
#SBATCH --gres=gpu:6               # Request 6 GPUs
#SBATCH --mem=128G                 # Job memory request
#SBATCH --time=200:00:00          # Time limit hrs:min:sec
#SBATCH --partition=gpu           # Partition (queue)

# Create necessary directories
mkdir -p logs plots trained_models results

# Activate conda environment
source activate shijia_env

# Check and install required packages
if ! pip list | grep -F pydicom > /dev/null; then
    echo "Installing pydicom..."
    pip install pydicom
fi

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# Run training script for all models
python train_models.py \
    --models convnext efficientnet resnet152  \
    --batch_size 32 \
    --num_epochs 30 \
    --learning_rate 1e-4 \
    --weight_decay 0.02 \
    --level_embeddings 256 \
    --freeze_backbone True \
    --unfreeze_last_n 32

# Alternative: Run models separately
# for MODEL in vit swin beit efficientnet resnet152 convnext; do
#     python main.py \
#         --models ${MODEL} \
#         --batch_size 32 \
#         --num_epochs 30 \
#         --learning_rate 1e-4 \
#         --weight_decay 0.01 \
#         --level_embeddings 128 \
#         --freeze_backbone True \
#         --unfreeze_last_n 20
# done