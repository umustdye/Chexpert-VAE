#!/bin/bash -l

#SBATCH --nodes=1 # Allocate *at least* 5 nodes to this job.
#SBATCH --ntasks=1 # Allocate *at most* 5 tasks for job steps in the job
#SBATCH --cpus-per-task=1 # Each task needs only one CPU
#SBATCH --mem=40G # This particular job won't need much memory
#SBATCH --time=17-02:01:00 # 17 days, 2 hours and 1 minute
#SBATCH --mail-user=hdye001@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="19 VAE Rerun"
#SBATCH -p gpu # You could pick other partitions for other jobs
#SBATCH --gpus=1
#SBATCH --wait-all-nodes=1 # Run once all resources are available
#SBATCH --output=output_%j-%N.txt # logging per job and per host in the current directory. Both stdout and stderr are logged.
# Place any commands you want to run below
hostname
date
nvidia-smi
/data/ChristianShelton/bin/runsing python ChexPhoto_VAE_Small_From_Memory.py
