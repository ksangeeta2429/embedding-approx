#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=mil-classify-ust
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=1-0
#SBATCH --mail-type=ALL
#SBATCH --output="mil-classify-ust-%j.out"
#SBATCH --err="mil-classify-ust-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`
module purge
module load cudnn/9.0v7.3.0.29
source activate l3embedding-tf-12-gpu

SRCDIR=.

## Begin user parameters
SONYC_UST_PATH=/path/to/sonyc-ust
EMBDIR=/path/to/embedding/directory
OUTPUTDIR=path/to/output/directory
NUM_HIDDEN=2
HIDDEN_SIZE=128
## End user parameters

ANNPATH=$SONYC_UST_PATH/annotations.csv
TAXPATH=$SONYC_UST_PATH/dcase-ust-taxonomy.yaml

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
