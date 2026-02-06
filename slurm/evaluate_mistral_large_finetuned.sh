#!/bin/bash

#SBATCH --partition=common
#SBATCH --job-name=eval_mistral_ft
#SBATCH --time=10:00:00

#SBATCH --account=ehpc-aif-2025pg01-226
#SBATCH --qos=ehpc-aif-2025pg01-226

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:6

#SBATCH -o eval_mistral_ft.%j.out
#SBATCH -e eval_mistral_ft.%j.err

cd /valhalla/projects/${SLURM_JOB_ACCOUNT} || exit 1

module purge || exit 1
module load anaconda3 || exit 1
module load nvidia/cuda/12 || exit 1

TARGET=/valhalla/projects/${SLURM_JOB_ACCOUNT}/virt_envs/torch
[ -d "${TARGET}" ] || { echo "Missing env: ${TARGET}"; exit 1; }

export PATH=$TARGET/bin:$PATH
export VIRTUAL_ENV=$TARGET

echo "Node: $(hostname)"
nvidia-smi -L

python code/evaluate_mistral_large_finetuned.py
