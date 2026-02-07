#!/bin/bash

#SBATCH --partition=common
#SBATCH --job-name=finetune_mistral_large
#SBATCH --time=24:00:00

#SBATCH --account=ehpc-aif-2025pg01-226
#SBATCH --qos=ehpc-aif-2025pg01-226

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:6

#SBATCH -o finetune_mistral_large.%j.out
#SBATCH -e finetune_mistral_large.%j.err

cd /valhalla/projects/${SLURM_JOB_ACCOUNT} || exit 1

module purge || exit 1
module load anaconda3 || exit 1
module load nvidia/cuda/12 || exit 1

TARGET_FOLDER=/valhalla/projects/${SLURM_JOB_ACCOUNT}/virt_envs/torch
[ -d "${TARGET_FOLDER}" ] || { echo "Missing env: ${TARGET_FOLDER}"; exit 1; }

export PATH="${TARGET_FOLDER}/bin:${PATH}"
export VIRTUAL_ENV="${TARGET_FOLDER}"

# CPU threads per process
export OMP_NUM_THREADS=4

# Debug info
echo "Node: $(hostname)"
nvidia-smi -L

# DDP with torchrun (8 GPUs in parallel)
MASTER_ADDR=$(hostname)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

torchrun \
    --standalone \
    --nproc_per_node=6 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    code/finetune/mistral/finetune_mistral_large.py
