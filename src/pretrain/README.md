# Pre-training

## Directories

* <code>[./config](./config)</code>: Contains configuration files.

## How to run

For all available arguments of the ```run_pretrain.py```, please refer to its help menu. The source code recycles the huggingface team's [```run_mlm.py```](https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py).
```
python3 run_pretrain.py --help
```

Instructions below use toy graph for demonstration. However, some arguments need to be adjusted for optimal training/evaluation. Please change the corresponding argument according to the following table.

|  Dataset  | --max_seq_length |
|:---------:|:----------------:|
|    FB13   |         -        |
| FB15K-237 |         -        |
|    WN11   |         -        |
|   WN18RR  |        188       |

Finally, note that ```--seed=530``` was used to ensure reproducibility.

### Without DeepSpeed

Running the following script will pre-train based on the pre-training data generated in the previous step. Also, update ```--per_device_train_batch_size``` to fit as much data as possible to your computer.
```
python3 run_pretrain.py \
	--do_train \
	--num_train_epochs=1 \
	--use_entity_relation_type \
	--entity_relation_type_mode=0 \
	--dataset_config_name=toy \
	--num_hops=2 \
	--model_name_or_path=roberta-large \
	--max_seq_length_percentile=0.999 \
	--output_dir=../../output/pretrain/toy \
	--per_device_train_batch_size=16 \
	--seed=530 \
	--save_steps=5000 \
	--save_total_limit=1 \
	--logging_first_step \
	--logging_steps=100 \
	--fp16
```

### With DeepSpeed

Using DeepSpeed (if you already installed it), you can run the ```run_pretrain.py``` script as follows:
```
deepspeed run_pretrain.py \
	--do_train \
	--adam_beta1=0.9 \
	--adam_beta2=0.98 \
	--adam_epsilon=1e-06 \
	--learning_rate=5e-05 \
	--max_grad_norm=0 \
	--weight_decay=0.01 \
	--warmup_steps=3100 \
	--num_train_epochs=1 \
	--use_entity_relation_type \
	--entity_relation_type_mode=1 \
	--dataset_config_name=toy \
	--num_hops=2 \
	--model_name_or_path=bert-base-cased \
	--max_seq_length_percentile=0.999 \
	--output_dir=../../output/pretrain/toy \
	--per_device_train_batch_size=16 \
	--seed=530 \
	--save_steps=5000 \
	--save_total_limit=1 \
	--logging_first_step \
	--logging_steps=100 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json

deepspeed run_pretrain.py \
	--do_train \
	--adam_beta1=0.9 \
	--adam_beta2=0.98 \
	--adam_epsilon=1e-06 \
	--learning_rate=5e-05 \
	--max_grad_norm=0 \
	--weight_decay=0.01 \
	--warmup_steps=3100 \
	--num_train_epochs=1 \
	--use_entity_relation_type \
	--entity_relation_type_mode=1 \
	--dataset_config_name=toy \
	--num_hops=2 \
	--model_name_or_path=roberta-large \
	--max_seq_length_percentile=0.999 \
	--output_dir=../../output/pretrain/toy \
	--per_device_train_batch_size=16 \
	--seed=530 \
	--save_steps=5000 \
	--save_total_limit=1 \
	--logging_first_step \
	--logging_steps=100 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json
```
