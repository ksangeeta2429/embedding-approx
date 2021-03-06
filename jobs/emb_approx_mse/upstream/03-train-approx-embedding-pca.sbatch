#!/usr/bin/env bash

#SBATCH --gres=gpu:2
#SBATCH --job-name=pca-sonyc-approx-mse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="pca-sonyc-approx-mse-%j.out"
#SBATCH --err="pca-sonyc-approx-mse-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
module load cudnn/9.0v7.3.0.29
source activate l3embedding-tf-12-gpu

SRCDIR=.
NUM_GPUS=2

### Begin user parameters
APPROX_MODE='pca'
LEARNING_RATE=0.0001
ASR=8000
NUM_MELS=64
HOP_LEN=160
NUM_DFT=1024
AUDIO_DIR=path/to/domain/audio/data
EMBEDDING_DIR=/path/to/transformed/teacher/embeddings/on/audio/data
OUTPUT_DIR=/path/to/output/dir
GOOGLE_DEV_APP_NAME='gsheet-name'
GSHEET_ID='gsheet-id-from-url'
### End user parameters

TRAIN_DATA_DIR=${AUDIO_DIR}/train
VAL_DATA_DIR=${AUDIO_DIR}/validate
EMBEDDING_TRAIN_DIR=${EMBEDDING_DIR}/train
EMBEDDING_VALID_DIR=${EMBEDDING_DIR}/validate

echo "APPROXIMATION=${APPROX_MODE}"
echo "TRAIN_DATA_DIR=${TRAIN_DATA_DIR}"
echo "EMBEDDING_TRAIN_DIR=${EMBEDDING_TRAIN_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

# CHOICE of learning-rate, train-epoch-size/steps-per-epoch and batch size
# Smaller learning rates will require more training epochs. Conversely, larger learning rates will require fewer training epochs.
# Smaller batch sizes are better suited to smaller learning rates given the noisy estimate of the error gradient.

# Typically, a grid search involves picking values approximately on a logarithmic scale, 
# e.g., a learning rate taken within the set {.1, .01, 10−3, 10−4 , 10−5}

python $SRCDIR/03_train_approx_embedding_mse.py \
	--num-epochs 300 \
	--train-epoch-size 4096 \
	--train-batch-size 64 \
	--model-type cnn_L3_melspec2 \
	--validation-epoch-size 1024 \
	--validation-batch-size 64 \
	--checkpoint-interval 10 \
	--gpus $NUM_GPUS \
	--approx-mode $APPROX_MODE \
	--samp-rate $ASR \
	--num-mels $NUM_MELS \
	--hop-length $HOP_LEN \
	--num-dft $NUM_DFT \
	--learning-rate $LEARNING_RATE \
	--random-state 20180216 \
	--gsheet-id $GSHEET_ID \
	--google-dev-app-name $GOOGLE_DEV_APP_NAME \
	--verbose \
	$TRAIN_DATA_DIR \
	$VAL_DATA_DIR \
	$EMBEDDING_TRAIN_DIR \
	$EMBEDDING_VALID_DIR \
	$OUTPUT_DIR
