deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=1e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=10 \
	--pretrain_checkpoint=final \
	--dataset_config_name=WN18RR \
	--model_name=roberta-large \
	--model_path=../../output/finetune/WN18RR \
	--model_type=sequence-classification \
	--model_subtype=forward-only \
	--max_seq_length=143 \
	--model_checkpoint=checkpoint-29850 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--max_eval_samples=50 \
	--output_dir=../../output/finetune/WN18RR \
	--per_device_eval_batch_size=4096 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee -a /data/Jason/log/WN18RR/roberta_noertype_hop1_epoch10/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-29850_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=1e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=10 \
	--pretrain_checkpoint=final \
	--dataset_config_name=WN18RR \
	--model_name=roberta-large \
	--model_path=../../output/finetune/WN18RR \
	--model_type=sequence-classification \
	--model_subtype=forward-only \
	--max_seq_length=143 \
	--model_checkpoint=checkpoint-59700 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--max_eval_samples=50 \
	--output_dir=../../output/finetune/WN18RR \
	--per_device_eval_batch_size=4096 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee -a /data/Jason/log/WN18RR/roberta_noertype_hop1_epoch10/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-59700_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=1e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=10 \
	--pretrain_checkpoint=final \
	--dataset_config_name=WN18RR \
	--model_name=roberta-large \
	--model_path=../../output/finetune/WN18RR \
	--model_type=sequence-classification \
	--model_subtype=forward-only \
	--max_seq_length=143 \
	--model_checkpoint=checkpoint-89550 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--max_eval_samples=50 \
	--output_dir=../../output/finetune/WN18RR \
	--per_device_eval_batch_size=4096 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee -a /data/Jason/log/WN18RR/roberta_noertype_hop1_epoch10/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-89550_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=1e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=10 \
	--pretrain_checkpoint=final \
	--dataset_config_name=WN18RR \
	--model_name=roberta-large \
	--model_path=../../output/finetune/WN18RR \
	--model_type=sequence-classification \
	--model_subtype=forward-only \
	--max_seq_length=143 \
	--model_checkpoint=checkpoint-119400 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--max_eval_samples=50 \
	--output_dir=../../output/finetune/WN18RR \
	--per_device_eval_batch_size=4096 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee -a /data/Jason/log/WN18RR/roberta_noertype_hop1_epoch10/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-119400_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=1e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=10 \
	--pretrain_checkpoint=final \
	--dataset_config_name=WN18RR \
	--model_name=roberta-large \
	--model_path=../../output/finetune/WN18RR \
	--model_type=sequence-classification \
	--model_subtype=forward-only \
	--max_seq_length=143 \
	--model_checkpoint=checkpoint-149250 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--max_eval_samples=50 \
	--output_dir=../../output/finetune/WN18RR \
	--per_device_eval_batch_size=4096 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee -a /data/Jason/log/WN18RR/roberta_noertype_hop1_epoch10/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-149250_wd0.01.log
