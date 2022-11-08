#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from multiprocessing import cpu_count
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
sys.path.append('..')

from datasets import load_dataset, Value, ClassLabel, Features  # noqa: E402
import pandas as pd  # noqa: E402
import torch.distributed  # noqa: E402
import transformers  # noqa: E402
from transformers import (  # noqa: E402
    MODEL_FOR_PRETRAINING_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForPreTraining,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process  # noqa: E402
from transformers.utils import check_min_version  # noqa: E402

from utils.preprocess_data import preprocess_data  # noqa: E402


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.6.0.dev0")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_PRETRAINING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
CONFIG_FOLDER = './config'
PROCESSED_DATA_DIR = {
    'FB15K-237': '../../data/FB15K-237/processed_data',
    'WN18RR': '../../data/WN18RR/processed_data',
    'UMLS': '../../data/UMLS/processed_data',
    'toy': '../../data/toy/processed_data',
}
MLM_FEATURES = {'path': Value('string')}
MLM_NSP_FEATURES = {
    'next_sentence_label': ClassLabel(names=['0', '1']),
    'path_1': Value('string'),
    'path_2': Value('string'),
}
PRETRAIN_FOLDER = 'pretrain'
ENTITIES_FILENAME = 'entities.txt'
RELATIONS_FILENAME = 'relations.txt'


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to pretrain.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization."
                          "Don't set if you want to train a model from scratch."},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model "
                          "type from the list: " + ", ".join(MODEL_TYPES)},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained "
                          "models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer "
                          "(backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use "
                          "(can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Will use the token generated when running `transformers-cli login`"
                          "(necessary to use this script with private models)."},
    )
    entity_relation_type_mode: int = field(
        default=None,
        metadata={"help": "Mode to use for entity/relatipn type IDs. "
                          "(0: all entities 0, forward relation 1, inverse relation 2.) "
                          "(1: all entities 0, each relation has unique id for both forward and inverse.)"},
    )

    def __post_init__(self):
        if self.entity_relation_type_mode and self.entity_relation_type_mode not in [0, 1]:
            raise ValueError(f'Invalid --entity_relation_type_mode: {self.entity_relation_type_mode}')


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use. "
                          f"({' | '.join(PROCESSED_DATA_DIR.keys())})"}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training set"}
    )
    max_seq_length: Optional[float] = field(
        default=None,
        metadata={"help": "Percentile (if < 1) or actual number for the maximum total input "
                          "sequence length after tokenization. Sequences longer than this "
                          "will be truncated."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=cpu_count()-1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, "
                          "truncate the number of training examples to this "
                          "value if set."},
    )
    num_hops: Optional[int] = field(
        default=None,
        metadata={"help": "Number of hops used for generating the pre-train data."},
    )

    def __post_init__(self):
        if self.dataset_config_name is None:
            raise ValueError("Dataset configuration name must be specified.")
        else:
            assert self.dataset_config_name in [x for x in PROCESSED_DATA_DIR.keys()], \
                f"Invalid dataset configuration name: {self.dataset_config_name}"

        if self.dataset_config_name == 'toy':
            logger.info('Updating --preprocessing_num_workers to 1 for toy dataset.')
            self.preprocessing_num_workers = 1

        if self.num_hops is None:
            raise ValueError("--num_hops must be specified.")


def main():
    ###################
    # parse arguments #
    ###################
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # For this script, we always train.
    if not training_args.do_train:
        raise ValueError("This script always runs training and --do_train must be set.")

    #################
    # setup logging #
    #################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, " +
        f"device: {training_args.device}, " +
        f"n_gpu: {training_args.n_gpu}, " +
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training parameters {training_args}")

    ################################################
    # update directories based on arguments passed #
    ################################################
    if model_args.entity_relation_type_mode:
        logger.info(f'Using entity/relation type IDs with mode {model_args.entity_relation_type_mode}.')
        training_args.output_dir += \
            f'_{model_args.model_name_or_path}_with_entity_relation_type_' \
            f'mode_{model_args.entity_relation_type_mode}'
    else:
        logger.info('Not using entity/relation type IDs.')
        training_args.output_dir += f'_{model_args.model_name_or_path}_without_entity_relation_type'

    training_args.output_dir += f'_hop-{data_args.num_hops}'
    training_args.output_dir += f'_epoch-{training_args.num_train_epochs}'

    training_args.logging_dir = os.path.join(training_args.output_dir, 'log')
    logger.info(f'Output directory updated: {training_args.output_dir}')
    logger.info(f'Logging directory updated: {training_args.logging_dir}')

    # Detecting last checkpoint.
    last_checkpoint = None
    if (os.path.isdir(training_args.output_dir) and
            training_args.do_train and not
            training_args.overwrite_output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. "
                "To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` "
                "to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    ###############################
    # load entities and relations #
    ###############################
    df_entities = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR[data_args.dataset_config_name], ENTITIES_FILENAME),
        sep='\t',
        na_filter=False)
    entity_lookup = dict(zip(df_entities['id'], df_entities['processed_name']))

    df_relations = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR[data_args.dataset_config_name], RELATIONS_FILENAME),
        sep='\t',
        na_filter=False)
    relation_lookup = dict(zip(df_relations['id'], df_relations['processed_name']))

    #######################################
    # load pretrained model and tokenizer #
    #######################################
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.model_name_or_path:
        model_name_or_path = model_args.model_name_or_path.replace('-scratch', '')
        pretrained_model_path = os.path.join(CONFIG_FOLDER, f'{model_name_or_path}.json')
        config = AutoConfig.from_pretrained(pretrained_model_path, **config_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. "
            "This is not supported by this script."
        )

    if model_args.entity_relation_type_mode == 0:
        assert config.entity_relation_type_vocab_size == 3
        logger.info('Keeping entity_relation_type_vocab_size from configuration file as is.')
    elif model_args.entity_relation_type_mode == 1:
        logger.info('Updating entity_relation_type_vocab_size from configuration.')
        config.update({'entity_relation_type_vocab_size': df_relations.shape[0]*2 + 1})
        logger.info(f'Configuration after update:\n{config}')

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.model_name_or_path:
        model_name_or_path = model_args.model_name_or_path.replace('-scratch', '')
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            config=config,
            **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. "
            "This is not supported by this script. "
        )

    if 'roberta-' in model_args.model_name_or_path:
        pretrain_mode = 'MLM'

        if 'scratch' not in model_args.model_name_or_path:
            model = AutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path.replace('-scratch', ''),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelForMaskedLM.from_config(config)
    elif 'bert-' in model_args.model_name_or_path:
        pretrain_mode = 'MLM-NSP'
        model = AutoModelForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        raise ValueError(f'Invalid model {model_args.model_name_or_path} specified!')

    model.resize_token_embeddings(len(tokenizer))

    ###################
    # get the dataset #
    ###################
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raise ValueError(
            "Downloading and loading a dataset from the hub "
            "is not yet supported by this script. "
        )
    else:
        data_filepath = os.path.join(
            PROCESSED_DATA_DIR[data_args.dataset_config_name],
            PRETRAIN_FOLDER,
            f'pretrain_{data_args.num_hops}-hops_{pretrain_mode}.txt'
        )
        features = Features(MLM_FEATURES) if pretrain_mode == 'MLM' else Features(MLM_NSP_FEATURES)

        if pretrain_mode == 'MLM':
            dataset = load_dataset(
                'text',
                data_files={'train': data_filepath},
                features=features,
                cache_dir=model_args.cache_dir,
            )['train']
        elif pretrain_mode == 'MLM-NSP':
            dataset = load_dataset(
                'csv',
                data_files={'train': data_filepath},
                features=features,
                cache_dir=model_args.cache_dir,
                delimiter='\t',
            )['train']
    logger.info(f'Dataset loaded: {dataset}')

    # # Block all processes other than the main. Use the main process to process data.
    # if training_args.local_rank > 0:
    #     print('Waiting for main process to perform the mapping')
    #     torch.distributed.barrier()

    # preprocess data
    # decode -> tokenize -> assign entity/relation type IDs -> truncate)
    processed_dataset = preprocess_data(
        dataset=dataset,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        entity_lookup=entity_lookup,
        relation_lookup=relation_lookup,
        entity_relation_type_mode=model_args.entity_relation_type_mode,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        model_max_length=tokenizer.model_max_length,
        model_name=model_args.model_name_or_path,
        is_finetune=False,
    )

    # # Let non-main processes resume and pickup the cached dataset.
    # if training_args.local_rank == 0:
    #     print('Loading results from the main process')
    #     torch.distributed.barrier()

    if data_args.max_train_samples is not None:
        processed_dataset = processed_dataset.select(range(data_args.max_train_samples))

    ######################
    # initialize trainer #
    ######################
    # Data collator takes care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    ############
    # Training #
    ############
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics

    max_train_samples = (
        data_args.max_train_samples
        if data_args.max_train_samples is not None else len(processed_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(processed_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
