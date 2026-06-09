#!/bin/bash
#SBATCH --job-name=auc-ensemble
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/NCAST/Output/submission-logs/inference/output/%j.out
#SBATCH -e /home/users/mendrika/NCAST/Output/submission-logs/inference/error/%j.err

echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Lead time: $1"
echo "Start time: $(date)"

LEAD_TIME=$1

module load jaspy/3.11
source /home/users/mendrika/virtual-env/DeepLearning/bin/activate

python /home/users/mendrika/NCAST/NCAST/model/evaluation/ensemble/compute-auc-ensemble.py \
    "${LEAD_TIME}"

echo "End time: $(date)"