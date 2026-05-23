#!/bin/bash
#SBATCH --job-name=apply-scaler
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/NCAST/Output/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/NCAST/Output/submission-logs/error/%j.err

set -e

module load jaspy/3.11
source /home/users/mendrika/SSA/bin/activate

export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

partition=$1
lead_time=$2

if [ -z "$partition" ] || [ -z "$lead_time" ]; then
    echo "Usage: sbatch apply-scaler.sh PARTITION LEAD_TIME"
    exit 1
fi

script=/home/users/mendrika/NCAST/NCAST/scaler/apply-scaler.py

if [ ! -f "$script" ]; then
    echo "Error: Python script not found at $script"
    exit 1
fi

python "$script" "$partition" "$lead_time"

echo "Job completed successfully."