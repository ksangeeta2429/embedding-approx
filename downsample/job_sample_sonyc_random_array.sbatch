#!/bin/bash

#SBATCH --job-name=sample-sonyc-random
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-0
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --output="sample-sonyc-random-%A-%a.out"
#SBATCH --err="sample-sonyc-random-%A-%a.err"

module purge
source ~/.bashrc
source activate sonyc-research-data

# Usage examples:
# sbatch --array=0-14 job_sample_sonyc_random_array.sbatch 200000 day
# sbatch --array=0-14 job_sample_sonyc_random_array.sbatch 200000 week

echo "SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
python -u sampling.py sample_sonyc_random $1 $2 $SLURM_ARRAY_TASK_ID
