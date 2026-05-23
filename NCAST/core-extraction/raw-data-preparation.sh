#!/bin/bash
#SBATCH --job-name=data-preparation
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/NCAST/Output/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/NCAST/Output/submission-logs/error/%j.err

set -e

module load jaspy/3.11
source /home/users/mendrika/virtual-env/DeepLearning/bin/activate

export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

year=$1
month=$2

if [ -z "$year" ] || [ -z "$month" ]; then
    echo "Usage: sbatch raw-data-preparation.sh YEAR MONTH"
    exit 1
fi

script=/home/users/mendrika/NCAST/NCAST/core-extraction/raw-data-preparation.py

if [ ! -f "$script" ]; then
    echo "Error: Python script not found at $script"
    exit 1
fi

python "$script" "$year" "$month"

echo "Job completed successfully."