"""
Todo:
    1. Allow user to set number of workers for multiprocessing.
    2. Use multiprocessing for post-processing test data?
"""
import argparse
from functools import partial
from multiprocessing import cpu_count, Pool
import os
from pathlib import Path
import random
import sys
from time import gmtime, strftime
from typing import List
sys.path.append('..')

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from utils import logging  # noqa: E402
from utils.utils import calc_chunksize, read_data  # noqa: E402
logger = logging.set_logging(__name__)

DATA_DIR = {
    'toy': '../../data/toy',
    'FB15K-237': '../../data/FB15K-237',
    'WN18RR': '../../data/WN18RR',
    'UMLS': '../../data/UMLS',
}
TASKS_MODEL_TYPES = {
    'link-prediction': ['sequence-classification', 'causal-language-modeling'],
    'triple-classification': ['sequence-classification'],
}
DATASET_WITH_POSITIVE_ONLY_TEST_TRIPLES = ['FB15K-237', 'toy', 'WN18RR', 'UMLS']
DEFAULT_VALIDATION_SAMPLES = -1
DEFAULT_CHUNKSIZE = None
DEFAULT_LOG_LEVEL = 'DEBUG'

# global variables
logger = None
_random = None
all_triples = None
all_triples_without_label = None
entities = None


def parse_argument() -> argparse.Namespace:
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Generate pretrain data.')

    parser.add_argument(
        '--dataset',
        required=True,
        help='Specify the dataset to generate pretraining data ({}).'.format(
            ' | '.join(DATA_DIR.keys())))

    parser.add_argument(
        '--task',
        required=True,
        help='Specify the downstream task ({}).'.format(' | '.join(TASKS_MODEL_TYPES.keys())))

    parser.add_argument(
        '--model_type',
        required=True,
        help='Specify the model type to use for each downstream task (--task) \
              ({}).'.format(TASKS_MODEL_TYPES))

    parser.add_argument(
        '--num_fine_tune_negative_train_samples',
        type=int,
        help='Number of negative training triples to sample for each positive triple. \
              Must be set for sequence-classification model type.')

    parser.add_argument(
        '--negative_train_corrupt_mode',
        type=str,
        help='Set mode for corrupt either or both heads or/and tails (corrupt-both | corrupt-one).')

    parser.add_argument(
        '--chunksize',
        default=DEFAULT_CHUNKSIZE,
        type=int,
        help='Chunksize to send to submit as a task. Default: {}.'.format(
            DEFAULT_CHUNKSIZE))

    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility.')

    parser.add_argument(
        '--log_level',
        default=DEFAULT_LOG_LEVEL,
        type=str,
        help='Set log level (DEBUG|INFO|WARNING|ERROR).')

    parser.add_argument(
        '--log_to_file',
        action='store_true',
        help='Set if you with to log to a file.')

    parser.add_argument(
        '--log_stat_file_dir',
        type=str,
        help='Set directory for saving logging/statistics files.')

    args = parser.parse_args()

    # Check integrity.
    if args.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
        raise ValueError(f'Invalid log level: {args.log_level}')

    if args.dataset not in DATA_DIR:
        raise ValueError(f'Invalid dataset specified: {args.dataset}')

    if args.task not in TASKS_MODEL_TYPES:
        raise ValueError(f'Invalid task: {args.task}')

    if args.model_type not in TASKS_MODEL_TYPES[args.task]:
        raise ValueError(f'Invalid model type for {args.task}: {args.model_type}')

    if args.model_type == 'sequence-classification' and not \
            args.num_fine_tune_negative_train_samples:
        raise ValueError('--num_fine_tune_negative_train_samples must be set')

    if args.negative_train_corrupt_mode not in ['corrupt-both', 'corrupt-one']:
        raise ValueError('Invalid --negative_train_corrupt_mode set')

    return args


def _sequence_classification(
        split: str,
        args: argparse.Namespace,
        triple: str,
        ) -> List[str]:
    head, relation, tail, label = triple.split(' ')
    forward = [f'{head} {relation} {tail}\t{label}']  # positve
    candidate_entities = list(set(entities) - {head, tail})

    def _get_corrupt_entity(h, r, t, where):
        random_idx = list(range(len(candidate_entities)))
        _random.shuffle(random_idx)

        if where == 'head':
            while random_idx:
                corrupt_entity = candidate_entities[random_idx.pop()]
                if f'{corrupt_entity} {r} {t}' not in all_triples_without_label:
                    break
        elif where == 'tail':
            while random_idx:
                corrupt_entity = candidate_entities[random_idx.pop()]
                if f'{h} {r} {corrupt_entity}' not in all_triples_without_label:
                    break

        if not random_idx:
            return None
        else:
            return corrupt_entity

    # negatives
    if split == 'train':
        assert label == '1', "Cannot generate negative training data if \
                              the input is already a negative data"

        if args.negative_train_corrupt_mode == 'corrupt-both':
            for _ in range(args.num_fine_tune_negative_train_samples):
                corrupt_head = _get_corrupt_entity(
                    head, relation, tail, 'head')
                if corrupt_head:
                    corrupt_triple = f'{corrupt_head} {relation} {tail}\t0'
                    forward.append(corrupt_triple)
            for _ in range(args.num_fine_tune_negative_train_samples):
                corrupt_tail = _get_corrupt_entity(
                    head, relation, tail, 'tail')
                if corrupt_tail:
                    corrupt_triple = f'{head} {relation} {corrupt_tail}\t0'
                    forward.append(corrupt_triple)
        else:
            coin_flip = _random.uniform(0, 1)
            for _ in range(args.num_fine_tune_negative_train_samples):
                if coin_flip > 0.5:  # corrupt head
                    corrupt_head = _get_corrupt_entity(
                        head, relation, tail, 'head')
                    if corrupt_head:
                        corrupt_triple = f'{corrupt_head} {relation} {tail}\t0'
                        forward.append(corrupt_triple)
                else:  # corrupt tail
                    corrupt_tail = _get_corrupt_entity(
                        head, relation, tail, 'tail')
                    if corrupt_tail:
                        corrupt_triple = f'{head} {relation} {corrupt_tail}\t0'
                        forward.append(corrupt_triple)

    elif split in ['val', 'test'] and any(
            dataset == args.dataset
            for dataset in DATASET_WITH_POSITIVE_ONLY_TEST_TRIPLES):
        assert label == '1', "Dataset should not have any negative triples"

        # corrupt head
        for corrupt_head in candidate_entities:
            if f'{corrupt_head} {relation} {tail}' in all_triples_without_label:
                continue
            forward.append(f'{corrupt_head} {relation} {tail}\t0')

        forward.append(f'{head} {relation} {tail}\t{label}')

        # corrupt tail
        for corrupt_tail in candidate_entities:
            if f'{head} {relation} {corrupt_tail}' in all_triples_without_label:
                continue
            forward.append(f'{head} {relation} {corrupt_tail}\t0')
    else:
        raise ValueError(f'Invalid split: {split}')

    # process reverse
    reverse = []
    for line in forward:
        triple, label = line.split('\t')
        head, relation, tail = triple.split(' ')
        reverse.append(f'{tail} !{relation} {head}\t{label}')

    return (forward, reverse)


def generate_link_prediction_data(
        train_triples: List[str],
        val_triples: List[str],
        test_triples: List[str],
        args: argparse.Namespace,
        n_workers: int,
        save_dir: str,
        ) -> None:
    """
    Generate link prediction data.
    """
    if args.model_type == 'sequence-classification':
        # generate train data by just corrupting either head/tail with
        # args.num_fine_tune_negative_train_samples
        def _handle_split(
                filepath_forward_only,
                filepath_forward_and_reverse,
                split,
                triples):
            with open(filepath_forward_only, mode='w', encoding='utf-8') as _f1, \
                 open(filepath_forward_and_reverse, mode='w', encoding='utf-8') as _f2:

                logger.info(f'File opened for saving {split} data: {filepath_forward_only}')
                logger.info(f'File opened for saving {split} data: {filepath_forward_and_reverse}')

                # Write header.
                _f1.write('path\tlabel\n')
                _f2.write('path\tlabel\n')

                if args.chunksize is None:
                    chunksize = calc_chunksize(n_workers=n_workers, len_iterable=len(triples))
                else:
                    chunksize = args.chunksize
                logger.info(f'Chunk size: {chunksize}')

                with Pool(n_workers) as p:
                    for forward, reverse in list(tqdm(p.imap(
                                partial(_sequence_classification, split, args),
                                triples,
                                chunksize=chunksize),
                            total=len(triples))):
                        if forward:
                            _f1.write('\n'.join(forward)+'\n')
                            _f2.write('\n'.join(forward)+'\n')
                            _f2.write('\n'.join(reverse)+'\n')

            logger.info(f'Forward only {split} file saved: {filepath_forward_only}')
            logger.info(f'Forward and reverse {split} file saved: {filepath_forward_and_reverse}')

        save_dir = os.path.join(
            save_dir,
            'link-prediction',
            args.model_type,
            args.negative_train_corrupt_mode,
            'forward-only')

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        def _split_data(
                triples,
                input_filepath,
                output_filepath,
                max_num_positives,
                split):
            with open(input_filepath, mode='r', encoding='utf-8') as _f1:
                next(_f1)  # skip header

                line = _f1.readline()
                for idx, tt in enumerate(tqdm(triples)):
                    lines = [line]
                    assert tt == line[:-1].replace('\t', ' ')

                    num_positives = 0
                    for line in _f1:
                        if line[:-1].split('\t')[-1] == '1':
                            num_positives += 1

                        if num_positives == max_num_positives:
                            break

                        lines.append(line)

                    filepath = os.path.join(output_filepath, f'{split}_{idx}.txt')
                    with open(filepath, mode='w', encoding='utf-8') as _f2:
                        _f2.write('path\tlabel\n')
                        _f2.write(''.join(lines))

        #########
        # train #
        #########
        train_forward_only_filepath = os.path.join(
            save_dir, f'{args.num_fine_tune_negative_train_samples}-negatives_train.txt')
        train_forward_and_reverse_filepath = train_forward_only_filepath.replace(
            'forward-only', 'forward-and-reverse')
        Path(os.path.dirname(train_forward_only_filepath)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(train_forward_and_reverse_filepath)).mkdir(parents=True, exist_ok=True)
        _handle_split(
            train_forward_only_filepath,
            train_forward_and_reverse_filepath,
            'train', train_triples)

        # we need to update all_triples_without_label since we generated corrupt data in training
        global all_triples_without_label
        updated_train_triples = read_data(train_forward_and_reverse_filepath)
        updated_train_triples = [t.replace('\t', ' ') for t in updated_train_triples]
        updated_train_triples_without_label = {t.rsplit(' ', maxsplit=1)[0] for t in updated_train_triples}
        logger.info(f'Number of all triples without label: {len(all_triples_without_label)}')
        all_triples_without_label = all_triples_without_label | updated_train_triples_without_label
        logger.info(f'Number of all triples without label after update: {len(all_triples_without_label)}')

        #######
        # val #
        #######
        # we shuffle validation set because we may only validate on subset of data
        _random.shuffle(val_triples)

        val_forward_only_filepath = os.path.join(save_dir, 'val.txt')
        val_forward_and_reverse_filepath = val_forward_only_filepath.replace(
            'forward-only', 'forward-and-reverse')
        _handle_split(
            val_forward_only_filepath,
            val_forward_and_reverse_filepath,
            'val', val_triples)

        # post-process validation data
        logger.info('Post-processing forward only validation data')
        val_forward_only_cut_dir = os.path.join(
            os.path.dirname(val_forward_only_filepath),
            'val_cut')
        Path(val_forward_only_cut_dir).mkdir(parents=True, exist_ok=True)

        _split_data(
            val_triples,
            val_forward_only_filepath,
            val_forward_only_cut_dir,
            2, 'val')

        logger.info('Post-processing forward and reverse validation data')
        test_forward_and_reverse_cut_dir = os.path.join(
            os.path.dirname(val_forward_and_reverse_filepath),
            'val_cut')
        Path(test_forward_and_reverse_cut_dir).mkdir(parents=True, exist_ok=True)

        _split_data(
            val_triples,
            val_forward_and_reverse_filepath,
            test_forward_and_reverse_cut_dir,
            4, 'val')

        ########
        # test #
        ########
        test_forward_only_filepath = os.path.join(save_dir, 'test.txt')
        test_forward_and_reverse_filepath = test_forward_only_filepath.replace(
            'forward-only', 'forward-and-reverse')
        _handle_split(
            test_forward_only_filepath,
            test_forward_and_reverse_filepath,
            'test', test_triples)

        # post-process test data
        logger.info('Post-processing forward only test data')
        test_forward_only_cut_dir = os.path.join(
            os.path.dirname(test_forward_only_filepath),
            'test_cut')
        Path(test_forward_only_cut_dir).mkdir(parents=True, exist_ok=True)

        _split_data(
            test_triples,
            test_forward_only_filepath,
            test_forward_only_cut_dir,
            2, 'test')

        logger.info('Post-processing forward and reverse test data')
        test_forward_and_reverse_cut_dir = os.path.join(
            os.path.dirname(test_forward_and_reverse_filepath),
            'test_cut')
        Path(test_forward_and_reverse_cut_dir).mkdir(parents=True, exist_ok=True)

        _split_data(
            test_triples,
            test_forward_and_reverse_filepath,
            test_forward_and_reverse_cut_dir,
            4, 'test')


def main():
    args = parse_argument()

    # Global variables.
    global logger
    global all_triples
    global all_triples_without_label
    global entities

    # Set log.
    if args.log_to_file:
        cur_time = strftime('%Y-%m-%d_%H:%M:%S', gmtime())
        log_file = os.path.join(args.log_stat_file_dir, f'log/{__file__}_{cur_time}.log')
    else:
        log_file = None

    logger = logging.set_logging(__name__, log_file=log_file, log_level=args.log_level)
    logger.debug(args)
    logger.info(f'Processing dataset \'{args.dataset}\' at \'{DATA_DIR[args.dataset]}\'.')

    # Set random seed.
    global _random
    if args.seed is not None:
        logger.info(f'Random seed set to: {args.seed}')
        _random = random.Random(args.seed)
    else:
        logger.info('Random seed is not set.')
        _random = random.Random()

    # Check CPU count.
    # n_workers = cpu_count() - 1
    n_workers = 1
    logger.info(f'Using {n_workers} CPUs for multiprocessing.')

    # Init file paths.
    processed_data_dir = os.path.join(DATA_DIR[args.dataset], 'processed_data')
    logger.info(f'Processed data directory: {processed_data_dir}')

    save_dir = os.path.join(processed_data_dir, 'finetune')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f'File save directory: {save_dir}')

    # Load data.
    train_triples = read_data(os.path.join(processed_data_dir, 'train.txt'))
    val_triples = read_data(os.path.join(processed_data_dir, 'val.txt'))
    test_triples = read_data(os.path.join(processed_data_dir, 'test.txt'))

    train_triples = [t.replace('\t', ' ') for t in train_triples]
    val_triples = [t.replace('\t', ' ') for t in val_triples]
    test_triples = [t.replace('\t', ' ') for t in test_triples]

    all_triples = set(train_triples + val_triples + test_triples)
    all_triples_without_label = {t.rsplit(' ', maxsplit=1)[0] for t in all_triples}

    df_entities = pd.read_csv(
        os.path.join(processed_data_dir, 'entities.txt'),
        sep='\t',
        na_filter=False)
    entities = df_entities['id'].tolist()

    # generate requested data
    if args.task == 'link-prediction':
        generate_link_prediction_data(
            train_triples=train_triples,
            val_triples=val_triples,
            test_triples=test_triples,
            args=args,
            n_workers=n_workers,
            save_dir=save_dir,
        )

    elif args.task == 'triple-classification':
        raise NotImplementedError()


if __name__ == '__main__':
    main()
