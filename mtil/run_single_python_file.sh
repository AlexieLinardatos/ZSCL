#!/bin/bash
#SBATCH --job-name=plot
#SBATCH --time=00:20:00            # max time
#SBATCH --mem=32GB                # memory
#SBATCH --cpus-per-task=4        # number of CPU cores
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/alanz21/thesis/mtil/logs/%x-%j.out  # output log file


module load python/3.11.5
module load cuda/12.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install tqdm ftfy regex wilds pandas
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch

pip install matplotlib

cd /scratch/alanz21/thesis/mtil/

python -m src.plot \
    --tsne \
    --dataset DTD \
    --model-path ckpt/DTD_freeze/MNIST_freeze/EuroSAT_freeze/Aircraft_freeze/Aircraft.pth \
    --save-path ./thesis_results/tsne_plots/freeze_Aircraft_on_DTD.png \
    --device cuda