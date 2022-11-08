# Pre-processing

## How to run

For all available arguments of the pre-processing python scripts, please refer to their help menu.
```
python3 generate_pretrain_data.py --help
python3 generate_finetune_data.py --help
```

Also, note that we used ```--seed=530``` for all the python scripts below to ensure reproducibility.

### 1. Generate pre-training data.

Instructions here use a small [toy graph](../../data/toy) for demonstration purpose. Running the following script will generate the pre-training data for the specified dataset.
```
python3 generate_pretrain_data.py \
	--dataset=toy \
	--num_hops=1 \
	--mode=MLM \
	--seed=530
```

The python script reads the files under the dataset-specific ```processed_data``` folder (*e.g.* [processed_data](../../data/toy/processed_data)) folder and saves the output files to the ```processed_data/pretrain``` folder (*e.g.* [processed_data/pretrain](../../data/toy/processed_data/pretrain)) folder.

### 2. Generate fine-tuning data.

#### Link prediction (using sequence-classification)

Instructions here use a small [toy graph](../../data/toy) for demonstration purpose. Running the following script will generate the fine-tuning data for the specified dataset.
```
python3 generate_finetune_data.py \
	--dataset=toy \
	--task=link-prediction \
	--model_type=sequence-classification \
	--num_fine_tune_negative_train_samples=5 \
	--seed=530
```

The python script reads the files under the dataset-specific ```processed_data``` folder (*e.g.* [processed_data](../../data/toy/processed_data)) folder and saves the output files to the ```processed_data/finetune``` folder (*e.g.* [processed_data/finetune](../../data/toy/processed_data/finetune)) folder.
