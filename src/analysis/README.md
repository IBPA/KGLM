# Analysis

## How to run

### 1. Generate pre-training data.

Instructions here use a small [toy graph](../../data/toy) for demonstration purpose. Running the following script will generate the pre-training data for the specified dataset.
```
python3 generate_pretrain_data.py \
	--dataset=toy \
	--num_hops=2 \
	--log_stat_file_dir=../../output/preprocess/generate_pretrain_data \
	--seed=530


```

The python script reads the files under the dataset-specific ```processed_data``` folder (*e.g.* [processed_data](../../data/toy/processed_data)) folder and saves the output files to the ```processed_data/pretrain``` folder (*e.g.* [processed_data/pretrain](../../data/toy/processed_data/pretrain)) folder.
