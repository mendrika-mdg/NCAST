#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --exclude=gpuhost006,gpuhost015
#SBATCH -o /home/users/mendrika/Object-Based-LSTMConv/slurm/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/Object-Based-LSTMConv/slurm/submission-logs/error/%j.err

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
lead_time=$1
seed=$2

if [ -z "$lead_time" ] || [ -z "$seed" ]; then
    echo "Usage: sbatch ensemble.sh <lead_time> <seed>"
    exit 1
fi

# run training
echo "Starting distributed training for lead_time=${lead_time}, seed=${seed}"
torchrun --standalone --nproc_per_node=4 \
    /home/users/mendrika/NCAST/NCAST/model/base/ncast.py \
    "$lead_time" "$seed"

echo "Training completed at $(date)"