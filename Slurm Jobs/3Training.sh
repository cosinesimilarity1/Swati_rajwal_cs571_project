#!/bin/bash
#SBATCH --job-name=python_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=999987
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=python_job_%j.out
#SBATCH --error=python_job_%j.err

# Initialize Conda and activate environment
# source /home/srajwal/anaconda3/etc/profile.d/conda.sh
# conda activate /home/srajwal/anaconda3/envs/nlp_virtual_env
source /local/scratch/shared-directories/ssanet/mlembed/bin/activate
# source /local/scratch/shared-directories/ssanet/envs/test_swati_env

# Execute Python script
python /local/scratch/shared-directories/ssanet/swati_folder/nlp_project/3perform_vs_train_size.py