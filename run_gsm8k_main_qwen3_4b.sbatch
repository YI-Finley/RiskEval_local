#!/bin/bash
#SBATCH -J gsm8k_main_qwen3_4b       # Job name
#SBATCH -p short                     # Partition (A100 is in short partition)
#SBATCH -N 1                         # Number of nodes
#SBATCH -n 1                         # Number of tasks
#SBATCH --gres=gpu:a100:1            # Request 1 A100 GPU
#SBATCH --mem=80G                    # Memory
#SBATCH -c 8                         # CPU cores
#SBATCH -t 240:00:00                 # Max runtime 240 hours
#SBATCH -o slurm-%j.out              # Standard output
#SBATCH -e slurm-%j.err              # Standard error

# Load CUDA module
module load cuda/11.4

# Activate your Python environment
source ~/py312env/bin/activate

# Go to RiskEval directory
cd ~/RiskEval

# Run RiskEval with the gsm8k config (absolute path)
python -m src.riskeval.cli --config /home/comp/23481501/datasets/RiskEval/configs/gsm8k_main_qwen3_4b.toml

