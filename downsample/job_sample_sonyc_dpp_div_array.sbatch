#!/bin/bash

#SBATCH --job-name=sample-sonyc-dppdiv
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-0
#SBATCH --mem=750GB
#SBATCH --mail-type=ALL
#SBATCH --output="sample-sonyc-dppdiv-%A-%a.out"
#SBATCH --err="sample-sonyc-dppdiv-%A-%a.err"

# Echo full command
echo ----------------------------------------
echo sbatch --array=$SLURM_ARRAY_TASK_ID job_sample_sonyc_dpp_div_array.sbatch $1 $2
echo -e "----------------------------------------\n"

module purge
source ~/.bashrc
source activate sonyc-research-data

# Usage examples:
# sbatch --array=0-14 job_sample_sonyc_dpp_div_array.sbatch 200000 day
# sbatch --array=0-14 job_sample_sonyc_dpp_div_array.sbatch 200000 week

echo "SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
python -u sampling.py sample_sonyc_dpp_div $1 $2 $SLURM_ARRAY_TASK_ID
