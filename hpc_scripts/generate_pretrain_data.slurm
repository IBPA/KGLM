#!/bin/bash

# options for sbatch
#SBATCH --job-name=BERTwithKG # Job name
#SBATCH --nodes=1 # should never be anything other than 1
#SBATCH --ntasks=64 # number of cpus to use
#SBATCH --time=3:00:00 # Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
#SBATCH --mem=50G # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --partition=dev # cluster partition
#SBATCH --output=/share/taglab/Jason/log/stdout_%j.out # File to which STDOUT will be written
#SBATCH --error=/share/taglab/Jason/log/stderr_%j.err # File to which STDERR will be written

# load modules
module load conda_pkgs
source activate conda_pkgs-4.9.2

# run
cd ../src/preprocess
python3 generate_pretrain_data.py \
	--dataset=WN11 \
	--num_hops=2 \
	--output_dir=../../output/preprocess/generate_pretrain_data \
	--seed=530
