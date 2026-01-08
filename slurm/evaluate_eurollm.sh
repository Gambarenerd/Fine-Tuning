#!/bin/bash

#SBATCH --partition=common
#SBATCH --job-name=eval_eurollm
#SBATCH --time=01:00:00

#SBATCH --account=ehpc-aif-2025pg01-226
#SBATCH --qos=ehpc-aif-2025pg01-226

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH -o eval_eurollm.%j.out
#SBATCH -e eval_eurollm.%j.err

cd /valhalla/projects/${SLURM_JOB_ACCOUNT}

module purge || exit 1
module load anaconda3 || exit 1
module load nvidia/cuda/12 || exit 1

TARGET=/valhalla/projects/${SLURM_JOB_ACCOUNT}/virt_envs/torch
export PATH=$TARGET/bin:$PATH
export VIRTUAL_ENV=$TARGET

nvidia-smi -L

python code/evaluate_eurollm.py
