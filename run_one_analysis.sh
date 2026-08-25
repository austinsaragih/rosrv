#!/bin/bash
#SBATCH --job-name=ROSR_A
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# MIT Engaging modules: keep the same software stack as the extended run.
module purge
module load community-modules
module load gurobi/13.0.2
module load miniforge/25.11.0-0

source activate scnd_env
cd "$SLURM_SUBMIT_DIR"

# NOTE: logs/ must exist BEFORE sbatch is submitted.
mkdir -p results

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=True

ANALYSIS_NUMBER="${1:-}"
if [ -z "$ANALYSIS_NUMBER" ]; then
    echo "ERROR: No analysis number provided."
    echo "Usage: sbatch run_one_analysis.sh <analysis_number>"
    echo "Example: sbatch run_one_analysis.sh 5"
    exit 1
fi

if ! [[ "$ANALYSIS_NUMBER" =~ ^([1-9]|1[0-3])$ ]]; then
    echo "ERROR: analysis number must be 1-13."
    exit 1
fi

echo "==========================================="
echo "ROSR Analysis ${ANALYSIS_NUMBER}"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Directory   : $(pwd)"
echo "Python      : $(which python)"
echo "Gurobi CLI  : $(which gurobi_cl)"
echo "CPUs        : $SLURM_CPUS_PER_TASK"
echo "Started     : $(date)"
echo "==========================================="

python - <<'PY'
import gurobipy as gp
import numpy as np
import pandas as pd
print("gurobipy version:", gp.gurobi.version())
print("numpy version:", np.__version__)
print("pandas version:", pd.__version__)
PY

# evaluation_include_no_launch remains False because we intentionally do NOT
# pass --evaluation-include-no-launch.
# Successful point checkpoints are skipped automatically on a resubmission.
python -u all_analyses_engaging.py \
    --analysis "$ANALYSIS_NUMBER" \
    --analysis11-mode full \
    --serial \
    --threads "$SLURM_CPUS_PER_TASK" \
    --output-dir results \
    --fail-job-on-error

# Create the final analysis-level CSV from the point checkpoints.
python -u all_analyses_engaging.py \
    --analysis "$ANALYSIS_NUMBER" \
    --analysis11-mode full \
    --merge \
    --output-dir results

echo "==========================================="
echo "Finished Analysis ${ANALYSIS_NUMBER}"
echo "Finished    : $(date)"
echo "==========================================="
