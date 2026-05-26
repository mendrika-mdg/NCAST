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

echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs allocated: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi

source /home/users/mendrika/virtual-env/DeepLearning/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONHASHSEED=0

lead_time=$1
lr=$2
dropout_p=$3
pos_weight=$4
alpha=$5

if [ -z "$lead_time" ] || [ -z "$lr" ] || [ -z "$dropout_p" ] || [ -z "$pos_weight" ] || [ -z "$alpha" ]; then
    echo "Usage: sbatch train_ncast.sh <lead_time> <lr> <dropout_p> <pos_weight> <alpha>"
    exit 1
fi

echo "Starting distributed training"
echo "lead_time=${lead_time}"
echo "lr=${lr}"
echo "dropout_p=${dropout_p}"
echo "pos_weight=${pos_weight}"
echo "alpha=${alpha}"

torchrun --standalone --nproc_per_node=4 \
    /home/users/mendrika/NCAST/NCAST/model/hyperparam-tuning/ncast-opt.py \
    "$lead_time" \
    "$lr" \
    "$dropout_p" \
    "$pos_weight" \
    "$alpha"

echo "Training completed at $(date)"