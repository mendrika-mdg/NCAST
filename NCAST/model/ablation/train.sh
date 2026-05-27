#!/bin/bash
#SBATCH --job-name=ncast-training
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --exclude=gpuhost006,gpuhost013
#SBATCH -o /home/users/mendrika/NCAST/Output/submission-logs/training/output/%j.out
#SBATCH -e /home/users/mendrika/NCAST/Output/submission-logs/training/error/%j.err

# basic info
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs allocated: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi

# activate virtual environment
source /home/users/mendrika/virtual-env/DeepLearning/bin/activate

# torch ddp config
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONHASHSEED=0

# parameters
lead_time=1
seed=1998 

# run training
echo "Starting distributed training for lead_time=${lead_time}, seed=${seed}"
torchrun --standalone --nproc_per_node=4 \
    /home/users/mendrika/NCAST/NCAST/model/ncast.py \
    "$lead_time" "$seed"

echo "Training completed at $(date)"
