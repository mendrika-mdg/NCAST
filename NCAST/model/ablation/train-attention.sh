#!/bin/bash
#SBATCH --job-name=ncast-ablation
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --exclude=gpuhost006,gpuhost013,gpuhost015,gpuhost016
#SBATCH -o /home/users/mendrika/NCAST/Output/submission-logs/ablation/output/%j.out
#SBATCH -e /home/users/mendrika/NCAST/Output/submission-logs/ablation/error/%j.err

echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs allocated: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi

source /home/users/mendrika/virtual-env/DeepLearning/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONHASHSEED=0

lead_time=$1

echo "Starting distributed training for lead_time=${lead_time}"

torchrun --standalone --nproc_per_node=4 \
    /home/users/mendrika/NCAST/NCAST/model/ablation/attention.py \
    "$lead_time"

echo "Training completed at $(date)"