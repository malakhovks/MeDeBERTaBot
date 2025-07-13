#!/usr/bin/env bash
###############################################################################
# jobscript-cpu.sh  –  Fine-tune DeBERTa-v3-small on a 190-core Intel CPU node
# Uses Intel oneAPI + PyTorch + IPEX.  Produces: deberta-csv-exp/ + metrics.json
###############################################################################
#SBATCH --job-name=deberta-cpu-train
#SBATCH --partition=scit5        # ← CHANGE to your CPU partition/queue
#SBATCH --cpus-per-task=190         # full socket / node (logical cores)
#SBATCH --mem=128G                  # adjust to your node’s RAM
# #SBATCH --time=02-00:00:00          # 2 days; tweak as needed
#SBATCH --output=cpu-train.%j.log   # stdout+stderr → log file
##SBATCH --exclusive                # uncomment for full-node exclusivity
#-----------------------------------------------------------------------------#

set -euo pipefail
echo "========  Job ${SLURM_JOB_ID:-N/A} launched on $(hostname) at $(date)  ========"

########################
# 0. Software modules  #
########################
# Either load oneAPI as a module …
set +u
source /opt/intel/oneapi/setvars.sh
set -u

##############################
# 1. Python virtual-env / conda
##############################
# The env must contain: torch ≥2.1, intel-extension-for-pytorch, transformers, datasets, evaluate
source .env-deberta-gpu/bin/activate

##########################################
# 2. Thread & affinity settings (oneAPI) #
##########################################
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0                      # let idle threads sleep

###############################################
# 3. Clean up any problematic HF_* env vars   #
###############################################
unset HF_HUB_HEADERS HF_HUB_EXTRA_HEADERS HUGGINGFACE_HUB_HEADERS || true

########################
# 4. Launch fine-tuning #
########################
python cpu-train.py

echo "========  Job ${SLURM_JOB_ID:-N/A} finished at $(date)  ========"
echo "Artifacts:"
echo "  • Fine-tuned model → $(realpath deberta-csv-exp)"
echo "  • Metrics JSON     → $(realpath metrics.json)"
