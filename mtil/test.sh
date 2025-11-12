#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=00:10:00            # max time
#SBATCH --mem=24GB                # memory
#SBATCH --cpus-per-task=4        # number of CPU cores
#SBATCH --gres=gpu:1             # request 1 GPU
#SBATCH --output=/scratch/alanz21/thesis/mtil/logs/%x-%j.out  # output log file
#SBATCH --signal=USR1@60

source /home/alanz21/jobs/ZSCL/bin/activate

cd /scratch/alanz21/thesis/mtil

mkdir -p logs

srun python -m src.main \
    --test \
    --save \
    ckpt/test \
    --load ckpt/test/DTD.pth \
    --train-dataset=DTD \
    --eval-datasets=Flowers