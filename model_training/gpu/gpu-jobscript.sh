#!/usr/bin/env bash
###############################################################################
# jobscript-gpu.sh  –  Fine-tune DeBERTa-v3-small on 1× RTX 2080 Ti
# Produces: deberta-csv-final/  and  metrics.json
###############################################################################
#SBATCH --job-name=deberta-train
#SBATCH --partition=scit5ai           # GPU queue
#SBATCH --gres=gpu:1                  # allocate 1 GPU (2080 Ti)
#SBATCH --cpus-per-task=10            # dataloader + tokeniser workers
#SBATCH --mem=64G                     # system RAM
# #SBATCH --time=24:00:00               # adjust as needed
#SBATCH --output=gpu-train.%j.log         # stdout+stderr → slurm log file
#-----------------------------------------------------------------------------#
# If you need exclusive access to the node:
# #SBATCH --exclusive
#-----------------------------------------------------------------------------#

set -euo pipefail
echo "========  Job $SLURM_JOB_ID launched on $(hostname) at $(date)  ========"

# 1. Activate your virtual environment
source .env-deberta-gpu/bin/activate

# 2. Hard-remove any malformed HF_*HEADERS that break huggingface_hub
unset HF_HUB_HEADERS HF_HUB_EXTRA_HEADERS HUGGINGFACE_HUB_HEADERS || true

# 3. Launch the training script
python gpu-train-2080ti-oom-safe.py

echo "========  Job $SLURM_JOB_ID finished at $(date)  ========"
echo "Artifacts:"
echo "  - Fine-tuned model → $(realpath deberta-csv-final)"
echo "  - Metrics JSON     → $(realpath metrics.json)"