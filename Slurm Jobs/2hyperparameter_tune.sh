#!/bin/bash
#SBATCH --job-name=python_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=499987
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=python_job_%j.out
#SBATCH --error=python_job_%j.err

source /local/scratch/shared-directories/ssanet/mlembed/bin/activate

# Execute Python script
python /local/scratch/shared-directories/ssanet/swati_folder/nlp_project/2hyperparameter_tune.py