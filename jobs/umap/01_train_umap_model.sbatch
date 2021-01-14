#!/usr/bin/env bash

#SBATCH --job-name=gen-samp-umap-transf-ust
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=250GB
#SBATCH --time=1-0
#SBATCH --mail-type=ALL
#SBATCH --output="gen-samp-umap-transf-ust-%j.out"
#SBATCH --err="gen-samp-umap-transf-ust-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
conda activate sonyc-research-data

# Echo full command
echo ----------------------------------------
echo sbatch $0 $1 $2 $3 $4 $5 $6 $7
echo -e "----------------------------------------\n"

SRCDIR=.
DATASET=$1
BATCH_SIZE=$2
EMB_LEN=$3
NEIGHBORS=$4
MIN_DIST=$5
SAMP=$6
TIMESCALE=$7
METRIC=euclidean

echo BATCH_SIZE=$BATCH_SIZE
echo EMB_LEN=$EMB_LEN
echo NEIGHBORS=$NEIGHBORS
echo METRIC=$METRIC
echo MIN_DIST=$MIN_DIST

if([ "$DATASET" = "sonyc" ]); then
    DATA_DIR="/scratch/dr2915/sonyc_samples/${SAMP}/ndata=${BATCH_SIZE}/timescale=${TIMESCALE}"
    TRANSFORM_DATA_DIR="/scratch/dr2915/embeddings/features/sonyc_ust/l3/melSpec_20200304183233_48000_256_242_2048"
    OUTPUT_DIR="$SCRATCH/reduced_embeddings/sonyc_ust/umap/upstream=${DATASET}/${SAMP}/timescale=${TIMESCALE}/umap_ndata=${BATCH_SIZE}_emb=${EMB_LEN}_nbrs=${NEIGHBORS}_mindist=${MIN_DIST}_mtrc=euclidean/"
else
   echo "Error: First argument can only be sonyc"
   exit
fi

echo DATA_DIR=$DATA_DIR
echo TRANSFORM_DATA_DIR=$TRANSFORM_DATA_DIR
echo OUTPUT_DIR=$OUTPUT_DIR

NEIGHBORS_LIST=( $NEIGHBORS ) #( 500 1000 5000 )
METRIC_LIST=( 'euclidean' )
MIN_DIST_LIST=( $MIN_DIST )

python -u $SRCDIR/01_train_save_reduced_emb.py \
  --batch-size $BATCH_SIZE \
  --approx-mode umap \
  --transform-data-dir $TRANSFORM_DATA_DIR \
  --neighbors-list ${NEIGHBORS_LIST[*]} \
  --min-dist-list ${MIN_DIST_LIST[*]} \
  --metric-list ${METRIC_LIST[*]} \
  --random-state 20180216 \
  $DATA_DIR \
  $OUTPUT_DIR \
  $EMB_LEN
