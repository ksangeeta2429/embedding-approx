#!/usr/bin/env bash

#SBATCH --job-name=extract-pca-upstream-emb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=dgx1
#SBATCH --mem=125GB
#SBATCH --time=3:00:00
#SBATCH --mail-type=ALL
#SBATCH --output="extract-pca-sonyc-%A-%a.out"                                                                                                                             
#SBATCH --err="extract-pca-sonyc-%A-%a.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
conda activate rapids
#conda activate sonyc-research-data

SRCDIR=.
MODE=$1
BATCH_SIZE=$2
TIMESCALE=$3
EMBLEN=$4
KERNEL='linear'
SUBSET='train'
PARTITION=$SLURM_ARRAY_TASK_ID

# /scratch/sk7898/dim_reduction/models/pca/dpp/day/500000/pca_ndata=500000_emb=256_kernel=linear.sav
MODEL_PATH="$SCRATCH/dim_reduction/models/pca/${MODE}/${TIMESCALE}/${BATCH_SIZE}/${EMBLEN}/pca_ndata=${BATCH_SIZE}_emb=${EMBLEN}_kernel=${KERNEL}.sav"
echo MODEL_PATH=$MODEL_PATH

UPSTREAM_DATA_DIR="/scratch/sk7898/sonyc_30mil/${SUBSET}"
echo DATA_DIR=$UPSTREAM_DATA_DIR

OUTPUT_DIR="$SCRATCH/reduced_embeddings/sonyc/pca/${MODE}/${TIMESCALE}/${BATCH_SIZE}/pca_batch_${BATCH_SIZE}_len_${EMBLEN}_kernel_${KERNEL}/${SUBSET}"
echo OUTPUT_DIR=$OUTPUT_DIR

# sbatch --array=0-14 02_extract_upstream_pca_emb.sbatch dpp 500000 day 256

python -u $SRCDIR/02_extract_upstream_reduced_emb.py \
  --random-state 20180216 \
  --approx-mode 'pca' \
  --batch-size $BATCH_SIZE \
  --partition-num $PARTITION \
  $MODEL_PATH \
  $UPSTREAM_DATA_DIR \
  $OUTPUT_DIR \
