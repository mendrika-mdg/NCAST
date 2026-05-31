#!/bin/bash
#SBATCH --job-name=temporal-ablation
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
ablation_name=$2

echo "Starting temporal ablation"
echo "lead_time=${lead_time}"
echo "ablation=${ablation_name}"

torchrun --standalone --nproc_per_node=4 \
    /home/users/mendrika/NCAST/NCAST/model/ablation/timestep.py \
    "$lead_time" \
    "$ablation_name"

echo "Training completed at $(date)"