#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=classify-ust-umap-mil
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=1-0
#SBATCH --mail-type=FAIL
#SBATCH --output="classify-ust-umap-mil-%j.out"
#SBATCH --err="classify-ust-umap-mil-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

if [ $USER == "sk7898" ]; then
    conda activate l3embedding-tf-12-gpu
else
    conda activate l3embedding-new-cpu
fi

module purge
module load cudnn/9.0v7.3.0.29

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

SRCDIR=.
ANNPATH=/beegfs/dr2915/sonyc_ust/data/annotations.csv
TAXPATH=/beegfs/dr2915/sonyc_ust/data/dcase-ust-taxonomy.yaml
EMBDIR="$SCRATCH/reduced_embeddings/sonyc_ust/umap/upstream=${DATASET}/${SAMP}/timescale=${TIMESCALE}/umap_ndata=${BATCH_SIZE}_emb=${EMB_LEN}_nbrs=${NEIGHBORS}_mindist=${MIN_DIST}_mtrc=euclidean"
OUTDIR="$SCRATCH/embeddings/classifier/sonyc_ust/mil/umap/upstream=sonyc/${SAMP}/timescale=${TIMESCALE}"

#module purge
#module load sox/intel/14.4.2
#module load ffmpeg/intel/3.2.2
#module load rubberband/intel/1.8.1

if [ $# -gt 7 ]; then
    NUM_HIDDEN=$8
    HIDDEN_SIZE=$9
else
    NUM_HIDDEN=0
    HIDDEN_SIZE=0
fi

EXPDIR="$OUTDIR/umap_ndata=${BATCH_SIZE}_emb=${EMB_LEN}_nbrs=${NEIGHBORS}_mindist=${MIN_DIST}_mtrc=euclidean/${NUM_HIDDEN}_${HIDDEN_SIZE}"
OUTPUTDIR=$EXPDIR/results

echo "NUM_HIDDEN=$NUM_HIDDEN"
echo "HIDDEN_SIZE=$HIDDEN_SIZE"

GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo'

python classifier/sonyc_ust/classify.py $ANNPATH $TAXPATH $EMBDIR $EXPDIR results \
    --hidden_layer_size $HIDDEN_SIZE \
    --num_hidden_layers $NUM_HIDDEN \
    --batch_size 64 \
    --num_epochs 1500 \
    --patience 1500 \
    --label_mode coarse \
    --target_mode mil \
    --no_timestamp \
    --optimizer adam

PYTHONPATH='.' python classifier/sonyc_ust/evaluate_predictions.py $OUTPUTDIR/output.csv $ANNPATH $TAXPATH $OUTPUTDIR $GSHEET_ID $GOOGLE_DEV_APP_NAME
