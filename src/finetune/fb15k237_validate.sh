# mode 1 lr 1e-6

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=1e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=1 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-187080 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode1_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-187080_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=1e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=1 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-374160 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode1_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-374160_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=1e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=1 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-561240 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode1_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-561240_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=1e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=1 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-748320 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode1_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-748320_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=1e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=1 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-935400 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode1_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-935400_wd0.01.log


# mode 1 lr 2e-6

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=2e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=1 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-187080 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode1_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr2e-06_checkpoint-187080_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=2e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=1 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-374160 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode1_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr2e-06_checkpoint-374160_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=2e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=1 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-561240 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode1_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr2e-06_checkpoint-561240_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=2e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=1 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-748320 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode1_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr2e-06_checkpoint-748320_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=2e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=1 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-935400 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode1_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr2e-06_checkpoint-935400_wd0.01.log






# mode 0 lr 1e-6

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=1e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=0 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-187080 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode0_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-187080_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=1e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=0 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-374160 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode0_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-374160_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=1e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=0 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-561240 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode0_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-561240_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=1e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=0 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-748320 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode0_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-748320_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=1e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=0 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-935400 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode0_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr1e-06_checkpoint-935400_wd0.01.log


# mode 0 lr 2e-6

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=2e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=0 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-187080 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode0_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr2e-06_checkpoint-187080_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=2e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=0 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-374160 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode0_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr2e-06_checkpoint-374160_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=2e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=0 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-561240 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode0_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr2e-06_checkpoint-561240_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=2e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=0 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-748320 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode0_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr2e-06_checkpoint-748320_wd0.01.log

deepspeed run_link_prediction.py \
	--do_eval \
	--learning_rate=2e-06 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--per_device_train_batch_size=32 \
	--use_entity_relation_type \
	--entity_relation_type_mode=0 \
	--pretrain_num_hops=1 \
	--pretrain_num_epochs=20 \
	--pretrain_checkpoint=final \
	--dataset_config_name=FB15K-237 \
	--model_name=roberta-large \
	--model_path=../../output/finetune/FB15K-237 \
	--model_type=sequence-classification \
	--model_subtype=forward-and-reverse \
	--max_seq_length=128 \
	--model_checkpoint=checkpoint-935400 \
	--num_fine_tune_negative_train_samples=5 \
	--negative_train_corrupt_mode=corrupt-both \
	--output_dir=../../output/finetune/FB15K-237 \
	--per_device_eval_batch_size=4096 \
	--max_eval_samples=60 \
	--fp16 \
	--deepspeed=./config/ds_config_zero2.json \
	--cache_dir=/data/Jason/.cache \
	| tee /data/Jason/log/FB15k-237/roberta_mode0_hop1_epoch20/corrupt-both/finetune_epoch5/finetune_validate_bs32_lr2e-06_checkpoint-935400_wd0.01.log
