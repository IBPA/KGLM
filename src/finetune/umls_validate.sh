# deepspeed run_link_prediction.py \
# 	--do_eval \
# 	--learning_rate=1e-06 \
# 	--weight_decay=0.01 \
# 	--num_train_epochs=5 \
# 	--per_device_train_batch_size=32 \
# 	--pretrain_num_hops=1 \
# 	--pretrain_num_epochs=20 \
# 	--entity_relation_type_mode=1 \
# 	--pretrain_checkpoint=final \
# 	--dataset_config_name=UMLS \
# 	--model_name=roberta-large \
# 	--model_path=../../output/finetune/UMLS \
# 	--model_type=sequence-classification \
# 	--model_subtype=forward-and-reverse \
# 	--max_seq_length=1.0 \
# 	--model_checkpoint=checkpoint-17585 \
# 	--num_fine_tune_negative_train_samples=5 \
# 	--negative_train_corrupt_mode=corrupt-both \
# 	--max_eval_samples=50 \
# 	--output_dir=../../output/finetune/UMLS \
# 	--per_device_eval_batch_size=16384 \
# 	--fp16 \
# 	--deepspeed=./config/ds_config_zero2.json \
# 	--cache_dir=/data/Jason/.cache \
# 	| tee -a /data/Jason/log/UMLS/roberta_mode1_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-17585_wd0.01.log

# deepspeed run_link_prediction.py \
# 	--do_eval \
# 	--learning_rate=1e-06 \
# 	--weight_decay=0.01 \
# 	--num_train_epochs=5 \
# 	--per_device_train_batch_size=32 \
# 	--pretrain_num_hops=1 \
# 	--pretrain_num_epochs=20 \
# 	--entity_relation_type_mode=1 \
# 	--pretrain_checkpoint=final \
# 	--dataset_config_name=UMLS \
# 	--model_name=roberta-large \
# 	--model_path=../../output/finetune/UMLS \
# 	--model_type=sequence-classification \
# 	--model_subtype=forward-and-reverse \
# 	--max_seq_length=1.0 \
# 	--model_checkpoint=checkpoint-14068 \
# 	--num_fine_tune_negative_train_samples=5 \
# 	--negative_train_corrupt_mode=corrupt-both \
# 	--max_eval_samples=50 \
# 	--output_dir=../../output/finetune/UMLS \
# 	--per_device_eval_batch_size=16384 \
# 	--fp16 \
# 	--deepspeed=./config/ds_config_zero2.json \
# 	--cache_dir=/data/Jason/.cache \
# 	| tee -a /data/Jason/log/UMLS/roberta_mode1_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-14068_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=1e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--entity_relation_type_mode=1 \
	--pretrain_checkpoint=final \
	--dataset_config_name=UMLS \
	--model_name=roberta-large \
	--model_path=../../output/finetune/UMLS \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=1.0 \
	--model_checkpoint=checkpoint-10551 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--max_eval_samples=50 \
	--output_dir=../../output/finetune/UMLS \
	--per_device_eval_batch_size=16384 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee -a /data/Jason/log/UMLS/roberta_mode1_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-10551_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=1e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--entity_relation_type_mode=1 \
	--pretrain_checkpoint=final \
	--dataset_config_name=UMLS \
	--model_name=roberta-large \
	--model_path=../../output/finetune/UMLS \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=1.0 \
	--model_checkpoint=checkpoint-7034 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--max_eval_samples=50 \
	--output_dir=../../output/finetune/UMLS \
	--per_device_eval_batch_size=16384 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee -a /data/Jason/log/UMLS/roberta_mode1_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-7034_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=1e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--entity_relation_type_mode=1 \
	--pretrain_checkpoint=final \
	--dataset_config_name=UMLS \
	--model_name=roberta-large \
	--model_path=../../output/finetune/UMLS \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=1.0 \
	--model_checkpoint=checkpoint-3517 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--max_eval_samples=50 \
	--output_dir=../../output/finetune/UMLS \
	--per_device_eval_batch_size=16384 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee -a /data/Jason/log/UMLS/roberta_mode1_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-3517_wd0.01.log
