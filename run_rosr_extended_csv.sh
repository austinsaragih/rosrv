#!/bin/bash
#SBATCH --job-name=ROSR_EXT_CSV
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --error=logs/%A_%a_%x.err
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-4%1

set -euo pipefail

module purge
module load community-modules
module load gurobi/13.0.2
module load miniforge/25.11.0-0

source activate scnd_env
cd "$SLURM_SUBMIT_DIR"

# NOTE: logs/ must exist BEFORE sbatch is submitted.
mkdir -p rosr_extended_outputs

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=True

CASE_NUMBER="$SLURM_ARRAY_TASK_ID"

echo "============================================================"
echo "ROSR extended CSV-only run"
echo "Case          : $CASE_NUMBER"
echo "Job ID        : $SLURM_JOB_ID"
echo "Array job ID  : $SLURM_ARRAY_JOB_ID"
echo "Node          : $(hostname)"
echo "Working dir   : $(pwd)"
echo "Python        : $(which python)"
echo "Gurobi CLI    : $(which gurobi_cl)"
echo "CPUs          : $SLURM_CPUS_PER_TASK"
echo "Started       : $(date)"
echo "============================================================"

python - <<'PY'
import gurobipy as gp
import numpy as np
import pandas as pd
print("gurobipy version:", gp.gurobi.version())
print("numpy version:", np.__version__)
print("pandas version:", pd.__version__)
PY

# The Python driver explicitly uses evaluation_include_no_launch=False.
python -u rosr_extended_csv_engaging.py \
    --case "$CASE_NUMBER" \
    --threads "$SLURM_CPUS_PER_TASK" \
    --output-root rosr_extended_outputs

echo "============================================================"
echo "Case $CASE_NUMBER finished: $(date)"
echo "============================================================"
