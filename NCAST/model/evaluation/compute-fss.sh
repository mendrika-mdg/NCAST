#!/bin/bash
#SBATCH --job-name=fss-eval
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/NCAST/Output/submission-logs/inference/output/%j.out
#SBATCH -e /home/users/mendrika/NCAST/Output/submission-logs/inference/error/%j.err

echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"

MODEL_NAME=$1
LEAD_TIME=$2

module load jaspy/3.11
source /home/users/mendrika/virtual-env/DeepLearning/bin/activate

python /home/users/mendrika/NCAST/NCAST/model/evaluation/compute-fss.py \
    "${MODEL_NAME}" \
    "${LEAD_TIME}"

echo "End time: $(date)"