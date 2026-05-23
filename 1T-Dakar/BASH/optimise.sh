#!/bin/bash
#SBATCH --job-name=Opt
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/NCAST/Output/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/NCAST/Output/submission-logs/error/%j.err

set -e
module load jaspy/3.11


export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

lead_time=$1

script=/home/users/mendrika/NCAST/1T-Dakar/PY/optimise_RF.py

if [ ! -f "$script" ]; then
    echo "Error: Python script not found at $script"
    exit 1
fi

echo "Running optimisation for lead time ${lead_time}h"

python -u "$script" "$lead_time"

echo "Finished successfully"