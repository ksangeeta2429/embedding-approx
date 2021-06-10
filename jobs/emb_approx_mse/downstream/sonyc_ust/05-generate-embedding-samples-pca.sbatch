#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=extract-reduced-emb-sonyc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2-0
#SBATCH --mail-type=ALL
#SBATCH --output="extract-reduced-emb-sonyc-%A-%a.out"
#SBATCH --err="extract-reduced-emb-sonyc-%A-%a.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
module load cudnn/9.0v7.3.0.29
source activate l3embedding-tf-12-gpu

## Begin user parameters
ASR=8000
NUM_MELS=64
HOP_LEN=160
NUM_DFT=1024

echo "ASR=$ASR"
echo "NUM_MELS=$NUM_MELS"
echo "HOP_LENGTH=$HOP_LEN"
echo "DFT=$NUM_DFT"

SRCDIR=.
L3_MODEL_PATH=/path/to/KD/student/models/model_best_valid_loss.h5
SONYC_UST_PATH=/path/to/sonyc-ust
DATA_DIR=$SONYC_UST_PATH/data
METADATA_PATH=$DATA_DIR/annotations.csv
OUTPUT_DIR=/path/to/output/directory
DATASET='sonyc_ust'

echo "WEIGHT FILE=$L3_MODEL_PATH"

echo "No explicit FMAX"
python -u $SRCDIR/05_generate_embedding_samples.py \
    --random-state 20180302 \
    --verbose \
    --features 'l3' \
    --l3embedding-model-path $L3_MODEL_PATH \
    --hop-size 0.1 \
    --gpus 0 \
    --samp-rate $ASR \
    --num-mels $NUM_MELS \
    --mel-hop-length $HOP_LEN \
    --num-dft $NUM_DFT \
    --with-melSpec \
    --sonyc-ust-annotation-path $METADATA_PATH \
    $DATASET \
    $DATA_DIR \
    $OUTPUT_DIR

