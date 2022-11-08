# Fine-tuning

## Directories

* <code>[./config](./config)</code>: Contains configuration files.

## How to run

Note that we used ```--seed=530``` for all the python scripts below to ensure reproducibility.

### 1. Link prediction.

For all available arguments of the ```run_link_prediction.py```, please refer to its help menu. The source code recycles the huggingface team's [```run_mlm.py```](https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py).
```
python3 run_link_prediction.py --help
```

#### Using sequence classification.

Train
```
python3 run_link_prediction.py \
	--do_train \
	--adam_beta1=$adam_beta1 \
	--adam_beta2=$adam_beta2 \
	--adam_epsilon=$adam_epsilon \
	--learning_rate=$learning_rate \
	--max_grad_norm=$max_grad_norm \
	--weight_decay=$weight_decay \
	--warmup_ratio=0.06 \
	--num_train_epochs=$epochs \
	--use_entity_relation_type \
	--entity_relation_type_mode=$entity_relation_type_mode \
	--pretrain_num_hops=$pretrain_num_hops \
	--pretrain_num_epochs=$pretrain_num_epochs \
	--pretrain_checkpoint=$pretrain_checkpoint \
	--dataset_config_name=$dataset \
	--model_name=$model_name \
	--model_path=../../output/pretrain/$dataset \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=$max_seq_length \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=$negative_train_corrupt_mode \
	--output_dir=../../output/finetune/$dataset \
	--per_device_train_batch_size=$batch_size \
	--seed=530 \
	--save_strategy=epoch \
	--logging_first_step \
	--logging_steps=1000 \
	--fp16 \
	--cache_dir=/data/Jason/.cache
```

Validate
```
python3 run_link_prediction.py \
	--do_eval \
	--learning_rate=$learning_rate \
	--weight_decay=$weight_decay \
	--num_train_epochs=$epochs \
	--per_device_train_batch_size=$train_batch_size \
	--use_entity_relation_type \
	--entity_relation_type_mode=$entity_relation_type_mode \
	--pretrain_num_hops=$pretrain_num_hops \
	--pretrain_num_epochs=$pretrain_num_epochs \
	--pretrain_checkpoint=$pretrain_checkpoint \
	--dataset_config_name=$dataset \
	--model_name=$model_name \
	--model_path=../../output/finetune/$dataset \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=$max_seq_length \
	--max_eval_samples=30 \
	--model_checkpoint=$cp \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=$negative_train_corrupt_mode \
	--output_dir=../../output/finetune/$dataset \
	--per_device_eval_batch_size=$batch_size \
	--fp16 \
	--cache_dir=/data/Jason/.cache
```

Test
```
$python_or_deepspeed run_link_prediction.py \
	--do_predict \
	--learning_rate=$learning_rate \
	--weight_decay=$weight_decay \
	--num_train_epochs=$epochs \
	--per_device_train_batch_size=$train_batch_size \
	--use_entity_relation_type \
	--entity_relation_type_mode=$entity_relation_type_mode \
	--pretrain_num_hops=$pretrain_num_hops \
	--pretrain_num_epochs=$pretrain_num_epochs \
	--pretrain_checkpoint=$pretrain_checkpoint \
	--dataset_config_name=$dataset \
	--model_name=$model_name \
	--model_path=../../output/finetune/$dataset \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=$max_seq_length \
	--model_checkpoint=$checkpoint \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=$negative_train_corrupt_mode \
	--output_dir=../../output/finetune/$dataset \
	--per_device_eval_batch_size=$batch_size \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache
```
