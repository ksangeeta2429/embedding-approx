#!/usr/bin/env bash

#SBATCH --job-name=gen-samp-pca-transf-ust
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=50GB
#SBATCH --time=1-0
#SBATCH --mail-type=ALL
#SBATCH --output="gen-samp-pca-transf-ust-%j.out"
#SBATCH --err="gen-samp-pca-transf-ust-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
#conda activate sonyc-research-data
conda activate rapids

# Echo full command
echo ----------------------------------------
echo sbatch $0 $1 $2 $3 $4 $5 $6
echo -e "----------------------------------------\n"

SRCDIR=.
DATASET=$1
BATCH_SIZE=$2
EMB_LEN=$3
KERNEL=$4
SAMP=$5
TIMESCALE=$6

echo BATCH_SIZE=$BATCH_SIZE
echo EMB_LEN=$EMB_LEN
echo KERNEL=$KERNEL

if([ "$DATASET" = "sonyc" ]); then
    DATA_DIR="$SCRATCH/sonyc_samples/${SAMP}/ndata=${BATCH_SIZE}/timescale=${TIMESCALE}"
    TRANSFORM_DATA_DIR="$SCRATCH/embeddings/features/sonyc_ust/l3/melSpec_20200304183233_48000_256_242_2048"
    SAVE_MODEL_DIR="$SCRATCH/dim_reduction/models/pca/${SAMP}/${TIMESCALE}/${BATCH_SIZE}/${EMB_LEN}"
    OUTPUT_DIR="$SCRATCH/reduced_embeddings/sonyc_ust/pca/upstream=${DATASET}/${SAMP}/timescale=${TIMESCALE}/pca_ndata=${BATCH_SIZE}_emb=${EMB_LEN}_kernel=${KERNEL}/"
else
   echo "Error: First argument can only be sonyc"
   exit
fi

echo DATA_DIR=$DATA_DIR
echo TRANSFORM_DATA_DIR=$TRANSFORM_DATA_DIR
echo SAVE_MODEL_DIR=$SAVE_MODEL_DIR
echo OUTPUT_DIR=$OUTPUT_DIR

# sbatch 01_train_pca_model.sbatch sonyc 500000 256 linear dpp day

python -u $SRCDIR/01_train_save_reduced_emb.py \
  --batch-size $BATCH_SIZE \
  --approx-mode pca \
  --transform-data-dir $TRANSFORM_DATA_DIR \
  --save-model-dir $SAVE_MODEL_DIR \
  --pca-kernel $KERNEL \
  --random-state 20180216 \
  $DATA_DIR \
  $OUTPUT_DIR \
  $EMB_LEN
