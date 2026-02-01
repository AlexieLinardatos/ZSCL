#!/bin/bash
#SBATCH --job-name=zscl_dtd_continue
#SBATCH --time=02:00:00            # max time
#SBATCH --mem=32GB                # memory
#SBATCH --cpus-per-task=4        # number of CPU cores
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/alexie/logs/%x-%j.out
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

REPO_ROOT="$HOME/projects/def-fqureshi/alexie/ZSCL"
cd "$REPO_ROOT/mtil"

mkdir -p logs

TARGET_DATASET="DTD"

SAVE_PATH="ckpt/clean/gradient_zscl/trained/DTD_trained"

PREV_LOAD_PATH="ckpt/clean/5000_iter/zscl/trained/DTD.pth"
# PREV_GRADIENT_PATH="ckpt/clean/gradient_zscl/trained/DTD_trained/grad_DTD.pth"

mkdir -p ${SAVE_PATH}

MODEL_NAME="${TARGET_DATASET}.pth"

DATASETS="DTD,MNIST,EuroSAT,Flowers"

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
    --iterations 5000 \
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
    ${LOAD} \
    ${START_ITERATION}
