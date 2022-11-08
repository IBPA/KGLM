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

from dataclasses import dataclass, field
from datetime import timedelta
import logging
from multiprocessing import cpu_count
import os
from pathlib import Path
import sys
from time import time
from typing import Optional, List, Dict
sys.path.append('..')

from datasets import load_dataset, ClassLabel, Value, Features  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import transformers  # noqa: E402
import torch.distributed  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402
from transformers.utils import check_min_version  # noqa: E402
from utils.utils import read_data, save_pkl, load_pkl  # noqa: E402
from utils.preprocess_data import preprocess_data  # noqa: E402


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.7.0.dev0")

logger = logging.getLogger(__name__)
MODEL_TYPE_SUBTYPES = {
    'sequence-classification': ['forward-only', 'forward-and-reverse'],
    'causal-language-modeling': ['forward-only', 'forward-and-reverse'],
}
CONFIG_FOLDER = './config'
PROCESSED_DATA_DIR = {
    'FB15K-237': '../../data/FB15K-237/processed_data',
    'WN18RR': '../../data/WN18RR/processed_data',
    'UMLS': '../../data/UMLS/processed_data',
    'toy': '../../data/toy/processed_data',
}
FEATURES = {
    'label': ClassLabel(names=['0', '1']),
    'path': Value('string'),
}
FINETUNE_FOLDER = 'finetune'
FINETUNE_TASK = 'link-prediction'
ENTITIES_FILENAME = 'entities.txt'
RELATIONS_FILENAME = 'relations.txt'
TRAIN_FILENAME = 'train.txt'
VALIDATION_FILENAME = 'val.txt'
TEST_FILENAME = 'test.txt'
ENCODED_DATASET_FOLDER = 'encoded'
TEST_CUT_FOLDER = 'test_cut'
TEST_RESULTS_FOLDER = 'test_results'
VAL_CUT_FOLDER = 'val_cut'
VAL_RESULTS_FOLDER = 'val_results'


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "The model name."},
    )
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization."},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Specify which task to use for link prediction. "
                          "({})".format(' | '.join(list(MODEL_TYPE_SUBTYPES.keys())))}
    )
    model_subtype: Optional[str] = field(
        default=None,
        metadata={"help": "If using sequence-classification for model type (--model_type), "
                          "specify one of ({}). ".format(' | '.join(
                              list(MODEL_TYPE_SUBTYPES['sequence-classification']))) +
                          "If using causal-language-modeling, "
                          "specify one of ({}). ".format(' | '.join(
                              list(MODEL_TYPE_SUBTYPES['causal-language-modeling'])))}
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
        metadata={
            "help": "Will use the token generated when running `transformers-cli login`"
                    "(necessary to use this script with private models)."
        },
    )
    entity_relation_type_mode: int = field(
        default=None,
        metadata={"help": "Mode to use for entity/relatipn type IDs. "
                          "(0: all entities 0, forward relation 1, inverse relation 2.) "
                          "(1: all entities 0, each relation has unique id for both forward and inverse.)"},
    )
    pretrain_num_hops: int = field(
        default=None,
        metadata={"help": "Number of hops used for generating the pre-train data."},
    )
    pretrain_num_epochs: float = field(
        default=None,
        metadata={"help": "Number of pre-train epochs."},
    )
    pretrain_checkpoint: str = field(
        default=None,
        metadata={"help": "Specify checkpoint (checkpoint-xxxxx) if not using the final checkpoint. "
                          "Set to 'final' if using final model."}
    )

    def __post_init__(self):
        if self.model_name is None or \
           self.model_type is None or \
           self.model_subtype is None:
            raise ValueError(
                "--model_name, --model_type, --model_subtype must all be set.")

        assert self.model_type in MODEL_TYPE_SUBTYPES, \
            f"Invalid model type: {self.model_type}"

        assert self.model_subtype in MODEL_TYPE_SUBTYPES[self.model_type], \
            f"Invalid model subtype: {self.model_subtype}"

        if self.entity_relation_type_mode and self.entity_relation_type_mode not in [0, 1]:
            raise ValueError(f'Invalid --entity_relation_type_mode: {self.entity_relation_type_mode}')

        if self.pretrain_num_hops is None and self.model_path is not None:
            raise ValueError("--pretrain_num_hops must be specified.")

        if self.pretrain_num_epochs is None and self.model_path is not None:
            raise ValueError("--pretrain_num_epochs must be specified.")

        if self.pretrain_checkpoint is None and self.model_path is not None:
            raise ValueError("--pretrain_checkpoint must be specified.")

        # update model path based on input
        if self.model_path is None:
            self.model_path = self.model_name
        else:
            if self.entity_relation_type_mode:
                self.model_path += \
                    f'_{self.model_name}_with_entity_relation_type_' + \
                    f'mode_{self.entity_relation_type_mode}'
            else:
                self.model_path += f'_{self.model_name}_without_entity_relation_type'
            self.model_path += f'_hop-{self.pretrain_num_hops}'
            self.model_path += f'_epoch-{self.pretrain_num_epochs}'

            if self.pretrain_checkpoint != 'final' and 'finetune' not in self.model_path:
                self.model_path = os.path.join(self.model_path, self.pretrain_checkpoint)
            elif self.pretrain_checkpoint != 'final' and 'finetune' in self.model_path:
                self.model_path += f'_{self.pretrain_checkpoint}'

            assert Path(self.model_path).is_dir(), \
                f'Model path does not exist: {self.model_path}'


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
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
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[float] = field(
        default=None,
        metadata={"help": "Percentile (if < 1) or actual number for the maximum total input "
                          "sequence length after tokenization. Sequences longer than this "
                          "will be truncated."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=cpu_count(),
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, "
                          "truncate the number of training examples to this "
                          "value if set."}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate "
                          "the number of evaluation examples to this value if set."}
    )
    model_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Model checkpoint for hyperparameter search using validation set. "
                          "checkpoint-xxxxx for intermediate checkpoints or final for the final model."}
    )
    num_fine_tune_negative_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of negative training triples to sample for each "
                          "positive triple."}
    )
    negative_train_corrupt_mode: Optional[str] = field(
        default=None,
        metadata={"help": "Specify if you wish to use training triples with both "
                          "heads and tails corrupted (corrupt-both) or either one "
                          " of the two (corrupt-one)."}
    )

    def __post_init__(self):
        if self.dataset_config_name is None:
            raise ValueError("Dataset configuration name must be specified.")
        else:
            assert self.dataset_config_name in [x for x in PROCESSED_DATA_DIR.keys()], \
                f"Invalid dataset configuration name: {self.dataset_config_name}"

        if self.num_fine_tune_negative_train_samples is None:
            raise ValueError("--num_fine_tune_negative_train_samples must be set.")

        if self.negative_train_corrupt_mode not in ['corrupt-both', 'corrupt-one']:
            raise ValueError("--negative_train_corrupt_mode must be set.")

        if self.dataset_config_name == 'toy':
            logger.info('Updating --preprocessing_num_workers to 1 for toy dataset.')
            self.preprocessing_num_workers = 1

        # if self.model_checkpoint is None:
        #     raise ValueError('--model_checkpoint must be set.')


def calculate_metrics(
        scores: np.array,
        labels: np.array,
        ranks_head_or_tail: List[int],
        ranks: List[int],
        hits_head_or_tail: Dict[int, List[bool]],
        hits: Dict[int, List[int]]) -> None:
    assert scores.shape == labels.shape, 'scores and labels dimensions do not match'

    index_array = np.argsort(-scores)
    sorted_labels = labels[index_array].tolist()
    rank = sorted_labels.index(1) + 1

    ranks_head_or_tail.append(rank)
    ranks.append(rank)

    for n in list(hits_head_or_tail.keys()):
        if rank <= n:
            hits_head_or_tail[n].append(True)
            hits[n].append(True)
        else:
            hits_head_or_tail[n].append(False)
            hits[n].append(False)


def log_or_save_metrics(
        mode: str,
        ranks_head: List[int],
        ranks_tail: List[int],
        ranks: List[int],
        hits_head: Dict[int, List[bool]],
        hits_tail: Dict[int, List[bool]],
        hits: Dict[int, List[bool]],
        save_parent_dir: Optional[str] = None,
        save_filename: Optional[str] = 'results.txt') -> None:
    logger.info(f'ranks_head: {ranks_head}')
    logger.info(f'ranks_tail: {ranks_tail}')
    logger.info(f'ranks: {ranks}')
    logger.info(f'hits_head: {hits_head}')
    logger.info(f'hits_tail: {hits_tail}')
    logger.info(f'hits: {hits}')

    contents = f'MR head: {np.mean(ranks_head)}\n'
    contents += f'MR tail: {np.mean(ranks_tail)}\n'
    contents += f'MR: {np.mean(ranks)}\n'
    contents += '\n'
    contents += f'MRR head: {np.mean(1. / np.array(ranks_head))}\n'
    contents += f'MRR tail: {np.mean(1. / np.array(ranks_tail))}\n'
    contents += f'MRR: {np.mean(1. / np.array(ranks))}\n'
    contents += '\n'

    for n in list(hits.keys()):
        contents += f'Hits head @{n}: {np.mean(hits_head[n])}\n'
        contents += f'Hits tail @{n}: {np.mean(hits_tail[n])}\n'
        contents += f'Hits @{n}: {np.mean(hits[n])}\n'
        contents += '\n'

    header = 'Intermediate metrics:' if mode == 'log' else 'Final metrics:'
    logger.info(f'{header}\n{contents}')

    if mode == 'save':
        if save_parent_dir is None:
            raise ValueError('save_parent_dir not specified')

        save_path = os.path.join(save_parent_dir, save_filename)
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        with open(save_path, mode='w', encoding='utf-8') as file:
            file.write(contents)

        logger.info(f'Test results saved to: {save_path}')


def load_intermediate_results(
        parent_dir: str,
        max_seq_length: float,
        model_checkpoint: str,
        filename: str = 'intermediate_results.pkl'):
    filename = filename.replace('.pkl', f'_{model_checkpoint}_msl-{max_seq_length}.pkl')
    filepath = os.path.join(parent_dir, filename)

    if not Path(filepath).exists():
        logger.info('Previous intermediate result not found')
        return None

    intermediate_results = load_pkl(filepath)
    logger.info(f'Previous intermediate result found: {filepath}')
    logger.info(f'Intermediate results: {intermediate_results}')

    return intermediate_results


def save_intermediate_results(
        idx: int,
        max_seq_length: float,
        ranks_head,
        ranks_tail,
        ranks,
        hits_head,
        hits_tail,
        hits,
        parent_dir: str,
        model_checkpoint: str,
        filename: str = 'intermediate_results.pkl') -> None:
    intermediate_results = {
        'idx': idx,
        'ranks_head': ranks_head,
        'ranks_tail': ranks_tail,
        'ranks': ranks,
        'hits_head': hits_head,
        'hits_tail': hits_tail,
        'hits': hits,
    }

    filename = filename.replace('.pkl', f'_{model_checkpoint}_msl-{max_seq_length}.pkl')

    Path(parent_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(parent_dir, filename)
    save_pkl(intermediate_results, filepath)
    logger.info(f'Saving prediction checkpoint: {filepath}')


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

    # Keep train/eval separate from test.
    if training_args.do_train and (training_args.do_eval or training_args.do_predict):
        raise ValueError(
            'This script does not support training (--do_train) '
            'and testing (--do_predict) simultaneously.')

    if model_args.model_path != model_args.model_name:
        if (training_args.do_eval or training_args.do_predict) and 'finetune' not in model_args.model_path:
            raise ValueError(
                'You are trying to predict on a test set and it seems like '
                'the model path (--model_path) does not contain \'finetune\'.')

    #################
    # setup logging #
    #################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, " +
        f"device: {training_args.device}, " +
        f"n_gpu: {training_args.n_gpu}, " +
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    ################################################
    # update directories based on arguments passed #
    ################################################
    if model_args.entity_relation_type_mode:
        logger.info(f'Using entity/relation type IDs with mode {model_args.entity_relation_type_mode}.')
        training_args.output_dir += f'_{model_args.model_name}_with_entity_relation_type_mode_{model_args.entity_relation_type_mode}'
    else:
        logger.info('Not using entity/relation type IDs.')
        training_args.output_dir += f'_{model_args.model_name}_without_entity_relation_type'

    if model_args.model_path != model_args.model_name:
        training_args.output_dir += f'_hop-{model_args.pretrain_num_hops}'
        training_args.output_dir += f'_epoch-{model_args.pretrain_num_epochs}'
        if model_args.pretrain_checkpoint != 'final':
            training_args.output_dir += f'_{model_args.pretrain_checkpoint}'

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        FINETUNE_TASK,
        f'{model_args.model_type}_{data_args.num_fine_tune_negative_train_samples}' +
        f'-negatives_{data_args.negative_train_corrupt_mode}_{model_args.model_subtype}',
        f'epochs-{training_args.num_train_epochs}_lr-{training_args.learning_rate}' +
        f'_weight-decay-{training_args.weight_decay}' +
        f'_bs-{training_args.per_device_train_batch_size}',
    )

    if training_args.do_eval or training_args.do_predict:
        if model_args.model_path == model_args.model_name:
            model_args.model_path = training_args.output_dir
        else:
            model_args.model_path = os.path.join(
                model_args.model_path,
                FINETUNE_TASK,
                f'{model_args.model_type}_{data_args.num_fine_tune_negative_train_samples}' +
                f'-negatives_{data_args.negative_train_corrupt_mode}_{model_args.model_subtype}',
                f'epochs-{training_args.num_train_epochs}_lr-{training_args.learning_rate}' +
                f'_weight-decay-{training_args.weight_decay}' +
                f'_bs-{training_args.per_device_train_batch_size}',
            )

        if data_args.model_checkpoint == 'final':
            logger.info('Using the final model')
        else:
            logger.info(f'Using the model at checkpoint: {data_args.model_checkpoint}')
            model_args.model_path = os.path.join(
                model_args.model_path, data_args.model_checkpoint)

    training_args.logging_dir = os.path.join(training_args.output_dir, 'log')

    logger.info(f'Model path directory: {model_args.model_path}')
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
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
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

    model_configuration_path = os.path.join(
        CONFIG_FOLDER,
        f'{model_args.model_name}-{model_args.model_type}.json')
    config = AutoConfig.from_pretrained(model_configuration_path, **config_kwargs)

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

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_path,
        config=config,
        **tokenizer_kwargs)

    if model_args.model_type == 'sequence-classification':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif model_args.model_type == 'causal-language-modeling':
        raise NotImplementedError('Causal language modeling not implemented.')

    model.resize_token_embeddings(len(tokenizer))

    ###################
    # get the dataset #
    ###################
    if training_args.do_train:
        if data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raise NotImplementedError(
                "Downloading and loading a dataset from the hub "
                "is not yet supported by this script. "
            )
        else:
            data_files_dir = os.path.join(
                PROCESSED_DATA_DIR[data_args.dataset_config_name],
                FINETUNE_FOLDER,
                FINETUNE_TASK,
                model_args.model_type,
                data_args.negative_train_corrupt_mode,
                model_args.model_subtype,
            )

            data_files = {
                'train': f'{data_files_dir}/{data_args.num_fine_tune_negative_train_samples}-negatives_train.txt',
            }

            dataset = load_dataset(
                'csv',
                data_files=data_files,
                delimiter='\t',
                features=Features(FEATURES),
                cache_dir=model_args.cache_dir,
            )['train']
        logger.info(f'Dataset loaded: {dataset}')

    # # Block all processes other than the main. Use the main process to process data.
    # if training_args.do_train and training_args.local_rank > 0:
    #     logger.info('Waiting for main process to perform the mapping')
    #     torch.distributed.barrier()

    if training_args.do_train:
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
            model_name=model_args.model_name,
            is_finetune=True,
        )

    # # Let non-main processes resume and pickup the cached dataset.
    # if training_args.do_train and training_args.local_rank == 0:
    #     logger.info('Loading results from the main process')
    #     torch.distributed.barrier()

    if data_args.max_train_samples is not None and training_args.do_train:
        processed_dataset = processed_dataset.select(range(data_args.max_train_samples))

    ######################
    # initialize trainer #
    ######################
    # Data collator takes care of randomly masking the tokens.
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    ############
    # Training #
    ############
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
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

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name, "tags": "fill-mask"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)

    ###################
    # Validate / Test #
    ###################
    if training_args.do_eval or training_args.do_predict:
        if training_args.do_eval:
            mode = 'val'
            logger.info('Validating...')
        else:
            mode = 'test'
            logger.info('Testing...')

        if model_args.model_subtype == 'forward-and-reverse':
            alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        results_save_dir = os.path.join(
            training_args.output_dir,
            TEST_RESULTS_FOLDER if training_args.do_predict else VAL_RESULTS_FOLDER)
        logger.info(f'Results save directory: {results_save_dir}')

        # load data
        triples = read_data(os.path.join(
            PROCESSED_DATA_DIR[data_args.dataset_config_name],
            TEST_FILENAME if training_args.do_predict else VALIDATION_FILENAME))
        num_triples = len(triples)

        if data_args.max_eval_samples is not None:
            num_triples = data_args.max_eval_samples

        # metrics
        intermediate_results = load_intermediate_results(
            results_save_dir,
            data_args.max_seq_length,
            model_checkpoint=data_args.model_checkpoint
        )
        if intermediate_results:
            idx_to_resume = intermediate_results['idx']
            ranks_head = intermediate_results['ranks_head']
            ranks_tail = intermediate_results['ranks_tail']
            ranks = intermediate_results['ranks']
            hits_head = intermediate_results['hits_head']
            hits_tail = intermediate_results['hits_tail']
            hits = intermediate_results['hits']
        else:
            idx_to_resume = -1
            if model_args.model_subtype == 'forward-and-reverse':
                ranks_head = {a: [] for a in alphas}
                ranks_tail = {a: [] for a in alphas}
                ranks = {a: [] for a in alphas}
                hits_head = {a: {1: [], 3: [], 5: [], 10: []} for a in alphas}
                hits_tail = {a: {1: [], 3: [], 5: [], 10: []} for a in alphas}
                hits = {a: {1: [], 3: [], 5: [], 10: []} for a in alphas}
            else:
                ranks_head, ranks_tail, ranks = [], [], []
                hits_head, hits_tail, hits = [{1: [], 3: [], 5: [], 10: []} for _ in range(3)]

        # We have different ways to testing for each model subtype
        if model_args.model_type == 'sequence-classification':
            data_files_dir = os.path.join(
                PROCESSED_DATA_DIR[data_args.dataset_config_name],
                FINETUNE_FOLDER,
                FINETUNE_TASK,
                model_args.model_type,
                data_args.negative_train_corrupt_mode,
                model_args.model_subtype,
                TEST_CUT_FOLDER if training_args.do_predict else VAL_CUT_FOLDER,
            )

            start = time()
            for idx in range(num_triples):
                if idx <= idx_to_resume:
                    logger.info(f'Skipping: {idx}/{num_triples-1}')
                    continue

                logger.info(f'Progress: {idx}/{num_triples-1}')
                logger.info(f'Time elapsed: {timedelta(seconds=time() - start)}')

                filename = f'{data_files_dir}/{mode}_{idx}.txt'
                logger.info(f'Filename: {filename}')
                dataset = load_dataset(
                    'csv',
                    data_files={mode: filename},
                    delimiter='\t',
                    features=Features(FEATURES),
                    cache_dir=model_args.cache_dir,
                )[mode]
                logger.info(f'Dataset loaded: {dataset}')

                current_triple = dataset[0]['path']
                logger.info(f'Processing triple: {current_triple}')

                if training_args.local_rank > 0:
                    logger.info('Waiting for main process to perform the mapping')
                    torch.distributed.barrier()

                # preprocess dataset
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
                    model_name=model_args.model_name,
                    is_finetune=True,
                )

                if training_args.local_rank == 0:
                    logger.info('Loading results from the main process')
                    torch.distributed.barrier()

                # predict
                predictions, label_ids, _ = trainer.predict(processed_dataset)

                # evaluate
                if model_args.model_subtype == 'forward-and-reverse':
                    for alpha in alphas:
                        logger.info(f'Using forward and reverse score with alpha: {alpha}')
                        assert predictions.shape[0] % 2 == 0, \
                            f'Shape of forward-and-reverse {mode} dataset is not multiple of two.'

                        first_half_index = int(predictions.shape[0] / 2)
                        forward_predictions = predictions[:first_half_index]
                        reverse_predictions = predictions[first_half_index:]

                        predictions_combined = alpha*forward_predictions + (1-alpha)*reverse_predictions
                        label_ids_cut = label_ids[:first_half_index]

                        # this separates head from tail
                        head_tail_cutoff_idx = np.where(label_ids_cut == 1)[0][1]

                        # head
                        calculate_metrics(
                            scores=predictions_combined[:head_tail_cutoff_idx, 1],
                            labels=label_ids_cut[:head_tail_cutoff_idx],
                            ranks_head_or_tail=ranks_head[alpha],
                            ranks=ranks[alpha],
                            hits_head_or_tail=hits_head[alpha],
                            hits=hits[alpha],
                        )

                        # tail
                        calculate_metrics(
                            scores=predictions_combined[head_tail_cutoff_idx:, 1],
                            labels=label_ids_cut[head_tail_cutoff_idx:],
                            ranks_head_or_tail=ranks_tail[alpha],
                            ranks=ranks[alpha],
                            hits_head_or_tail=hits_tail[alpha],
                            hits=hits[alpha],
                        )

                        log_or_save_metrics(
                            mode='log',
                            ranks_head=ranks_head[alpha],
                            ranks_tail=ranks_tail[alpha],
                            ranks=ranks[alpha],
                            hits_head=hits_head[alpha],
                            hits_tail=hits_tail[alpha],
                            hits=hits[alpha])
                else:
                    # this separates head from tail
                    head_tail_cutoff_idx = np.where(label_ids == 1)[0][1]

                    # head
                    calculate_metrics(
                        scores=predictions[:head_tail_cutoff_idx, 1],
                        labels=label_ids[:head_tail_cutoff_idx],
                        ranks_head_or_tail=ranks_head,
                        ranks=ranks,
                        hits_head_or_tail=hits_head,
                        hits=hits,
                    )

                    # tail
                    calculate_metrics(
                        scores=predictions[head_tail_cutoff_idx:, 1],
                        labels=label_ids[head_tail_cutoff_idx:],
                        ranks_head_or_tail=ranks_tail,
                        ranks=ranks,
                        hits_head_or_tail=hits_tail,
                        hits=hits,
                    )

                    log_or_save_metrics(
                        mode='log',
                        ranks_head=ranks_head,
                        ranks_tail=ranks_tail,
                        ranks=ranks,
                        hits_head=hits_head,
                        hits_tail=hits_tail,
                        hits=hits)

                save_intermediate_results(
                    idx=idx,
                    max_seq_length=data_args.max_seq_length,
                    ranks_head=ranks_head,
                    ranks_tail=ranks_tail,
                    ranks=ranks,
                    hits_head=hits_head,
                    hits_tail=hits_tail,
                    hits=hits,
                    parent_dir=results_save_dir,
                    model_checkpoint=data_args.model_checkpoint)

            logger.info(f'{mode} complete.')

            # save results
            if model_args.model_subtype == 'forward-and-reverse':
                for alpha in alphas:
                    log_or_save_metrics(
                        mode='save',
                        ranks_head=ranks_head[alpha],
                        ranks_tail=ranks_tail[alpha],
                        ranks=ranks[alpha],
                        hits_head=hits_head[alpha],
                        hits_tail=hits_tail[alpha],
                        hits=hits[alpha],
                        save_parent_dir=results_save_dir,
                        save_filename=f'results_{data_args.model_checkpoint}_alpha-{alpha}.txt')
            else:
                log_or_save_metrics(
                    mode='save',
                    ranks_head=ranks_head,
                    ranks_tail=ranks_tail,
                    ranks=ranks,
                    hits_head=hits_head,
                    hits_tail=hits_tail,
                    hits=hits,
                    save_parent_dir=results_save_dir,
                    save_filename=f'results_{data_args.model_checkpoint}.txt')
        elif model_args.model_type == 'causal-language-modeling':
            raise NotImplementedError('Causal language modeling not implemented.')


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
