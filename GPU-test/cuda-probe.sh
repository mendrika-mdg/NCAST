#!/bin/bash
#SBATCH --job-name=cuda-test
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # One task per GPU
#SBATCH --gres=gpu:4                # 4 GPUs per node
#SBATCH --cpus-per-task=4           # Tune based on data loading needs
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclude=gpuhost006        # Avoid node with broken GPUs
#SBATCH -o /home/users/mendrika/NCAST/Output/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/NCAST/Output/submission-logs/error/%j.err

source /home/users/mendrika/virtual-env/DeepLearning/bin/activate

echo "Node: $(hostname)"
nvidia-smi

srun python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/gpu-check/run-test/cuda-probe.py