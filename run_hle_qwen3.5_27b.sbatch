#!/bin/bash
#SBATCH -J hle_qwen3.5_27b           # Job name
#SBATCH -p short                     # Partition (A100 is in short partition)
#SBATCH -N 1                         # Number of nodes
#SBATCH -n 1                         # Number of tasks
#SBATCH --gres=gpu:a100:2            # Request 2 A100 GPUs (80GB each)
#SBATCH --mem=300G                   # Memory
#SBATCH -c 16                        # CPU cores
#SBATCH -t 240:00:00                 # Max runtime 240 hours (24*10)
#SBATCH -o /home/datasets/23481501/RiskEval/output/slurm-%j.out   # Standard output
#SBATCH -e /home/datasets/23481501/RiskEval/output/slurm-%j.err   # Standard error

# Load CUDA module
module load cuda/11.4

# Activate your Python environment
source ~/py312env/bin/activate

# Go to RiskEval directory
cd /home/comp/23481501/datasets/RiskEval

# Run RiskEval with the HLE config (absolute path)
python -m src.riskeval.cli --config /home/comp/23481501/datasets/RiskEval/configs/hle_qwen3.5-27B_local.toml

