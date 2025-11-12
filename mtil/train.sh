#!/bin/bash
#SBATCH --job-name=cnn_train
#SBATCH --time=12:00:00          # max time
#SBATCH --mem=32G                # memory
#SBATCH --cpus-per-task=4        # number of CPU cores
#SBATCH --gres=gpu:1             # request 1 GPU
#SBATCH --output=logs/%x-%j.out  # output log file

module load python/3.10 pytorch/2.2 cuda/12.1

# cd ~/
srun python -m src.main \
    --train-mode=whole \
    --train-dataset=DTD \
    --lr=1e-5\
    --ls 0.2 \
    --iterations 1000 \
    --method ZSCL \
    --image_loss \
    --text_loss \
    --we \
    --avg_freq 100 \
    --l2 1 \
    --ref-dataset ImageNet \
    --ref-sentences conceptual_captions \
    --save ckpt/exp_000