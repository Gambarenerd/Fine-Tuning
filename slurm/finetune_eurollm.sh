#!/bin/bash

#SBATCH --partition=common
#SBATCH --job-name=finetune_eurollm_2gpu
#SBATCH --time=02:00:00

#SBATCH --account=ehpc-aif-2025pg01-226
#SBATCH --qos=ehpc-aif-2025pg01-226

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2

#SBATCH -o finetune_eurollm.%j.out
#SBATCH -e finetune_eurollm.%j.err

# Ensure that install.%j.out and install.%j.err are saved in the directory where you
# submit the job. Set the working directory of the Bash shell to the folder from
# which the script is launched.

cd /valhalla/projects/${SLURM_JOB_ACCOUNT} || exit 1

module purge || exit 1
module load anaconda3 || exit 1
module load nvidia/cuda/12 || exit 1

TARGET_FOLDER=/valhalla/projects/${SLURM_JOB_ACCOUNT}/virt_envs/torch
[ -d "${TARGET_FOLDER}" ] || { echo "Missing env: ${TARGET_FOLDER}"; exit 1; }

export PATH="${TARGET_FOLDER}/bin:${PATH}"
export VIRTUAL_ENV="${TARGET_FOLDER}"

# (Consigliato) thread CPU per processo
export OMP_NUM_THREADS=4

# (Opzionale ma utile per debug)
echo "Node: $(hostname)"
nvidia-smi -L

# Master address/port per torchrun (su singolo nodo basta hostname)
MASTER_ADDR=$(hostname)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

torchrun \
  --standalone \
  --nproc_per_node=2 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  code/finetune_eurollm.py
