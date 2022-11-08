#!/bin/bash

# options for sbatch
#SBATCH --job-name=BERTwithKG
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:2
#SBATCH --partition=production
#SBATCH --output=../output/pretrain/run_pretrain/log/stdout_%j.out
#SBATCH --error=../output/pretrain/run_pretrain/log/stderr_%j.err

# load modules
cd ..
source ./env/bin/activate

# for some reason, --cache_dir doesn't work well with run_pretrain.py
# export cache directories manually like this
export HF_MODULES_CACHE="/share/taglab/Jason/BERTwithKG/output/pretrain/.cache"
export HF_DATASETS_CACHE="/share/taglab/Jason/BERTwithKG/output/pretrain/.cache"
export TRANSFORMERS_CACHE="/share/taglab/Jason/BERTwithKG/output/pretrain/.cache"

# run
cd ./src/pretrain
python3 run_pretrain.py \
	--do_train \
	--dataset_config_name=WN18RR \
	--model_name_or_path=bert-base-uncased \
	--max_seq_length=512 \
	--output_dir=../../output/pretrain/WN18RR \
	--overwrite_output_dir \
	--logging_dir=../../output/pretrain/WN18RR/log \
	--per_device_train_batch_size=64 \
	--seed=530

deactivate
