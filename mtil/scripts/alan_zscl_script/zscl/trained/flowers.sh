#!/bin/bash
#SBATCH --job-name=gradient_zscl_flowers
#SBATCH --time=02:15:00            # max time
#SBATCH --mem=48GB                # memory
#SBATCH --cpus-per-task=4        # number of CPU cores
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/alanz21/thesis/mtil/logs/%x-%j.out  # output log file
#SBATCH --signal=USR1@60

nvidia-smi

module load python/3.11.5
module load cuda/12.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install tqdm ftfy regex wilds pandas
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch

cd /scratch/alanz21/thesis/mtil

mkdir -p logs


TARGET_DATASET="Flowers"

SAVE_PATH="ckpt/clean/gradient_zscl/trained/DTD_trained/MNIST_trained/EuroSAT_trained2/Flowers_trained2"

PREV_LOAD_PATH="ckpt/clean/gradient_zscl/trained/DTD_trained/MNIST_trained/EuroSAT_trained2/EuroSAT.pth"

PREV_GRADIENT_PATH="ckpt/clean/gradient_zscl/trained/DTD_trained/MNIST_trained/EuroSAT_trained2/grad_EuroSAT.pth"

mkdir -p ${SAVE_PATH}

MODEL_NAME="${TARGET_DATASET}.pth"

DATASETS="DTD,MNIST,EuroSAT,Flowers,ObjectNet"

#check if model exists
LOAD=""
START_ITERATION=""
if [ -f "${SAVE_PATH}/${MODEL_NAME}" ]; then
    LOAD="--load ${SAVE_PATH}/${MODEL_NAME}"
else
    LOAD="--load ${PREV_LOAD_PATH}"
    START_ITERATION="--start-iteration 0"
fi

srun python -m src.main \
    --train-mode=whole \
    --train-dataset=${TARGET_DATASET} \
    --lr=1e-5\
    --ls 0.2 \
    --iterations 2000 \
    --method ZSCL \
    --image_loss \
    --text_loss \
    --we \
    --avg_freq 100 \
    --l2 1 \
    --ref-dataset ImageNet \
    --ref-sentences conceptual_captions \
    --save ${SAVE_PATH} \
    --eval-datasets ${DATASETS} \
    --eval-interval 250 \
    --custom-finetune \
    --max-evaluation-size 500 \
    --orthogonal-gradients 10 \
    --orthogonal-gradients-path ${PREV_GRADIENT_PATH} \
    ${LOAD} \
    ${START_ITERATION}
