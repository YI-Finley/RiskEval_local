#!/bin/bash
#SBATCH -J hle_qwen3_4b_deepseek     # Job name
#SBATCH -p short                     # Partition (A100 is in short partition)
#SBATCH -N 1                         # Number of nodes
#SBATCH -n 1                         # Number of tasks
#SBATCH --gres=gpu:a100:1            # Request 1 A100 GPU
#SBATCH --mem=80G                    # Memory
#SBATCH -c 8                         # CPU cores
#SBATCH -t 336:00:00                 # Max runtime 336 hours
#SBATCH -o slurm-%j.out              # Standard output
#SBATCH -e slurm-%j.err              # Standard error

# Load CUDA module
module load cuda/11.4

# Activate your Python environment
source ~/py312env/bin/activate

# --- 环境修复部分 ---
echo "Checking and installing required dependencies..."
pip install --upgrade pip
pip install --upgrade transformers accelerate safetensors bitsandbytes

# Go to RiskEval directory
cd ~/RiskEval

# Run RiskEval with the HLE config (absolute path)
python -m src.riskeval.cli --config /home/comp/23481501/datasets/RiskEval/configs/hle_qwen3-4B_local.toml

