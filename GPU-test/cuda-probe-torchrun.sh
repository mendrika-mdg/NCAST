#!/bin/bash
#SBATCH --job-name=cuda-test
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # torchrun will spawn processes
#SBATCH --gres=gpu:4                # 4 GPUs per node
#SBATCH --cpus-per-task=4           # Tune based on data loading needs
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -o /home/users/mendrika/NCAST/Output/submission-logs/training/output/%j.out
#SBATCH -e /home/users/mendrika/NCAST/Output/submission-logs/training/error/%j.err

source /home/users/mendrika/virtual-env/DeepLearning/bin/activate

echo "Node: $(hostname)"
nvidia-smi

# Optional: pick a random port
export MASTER_PORT=$((12000 + RANDOM % 20000))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the CUDA probe with 4 distributed processes (1 per GPU)
torchrun --standalone --nproc_per_node=4 \
  /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/gpu-check/run-test/cuda-probe.py
