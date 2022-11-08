"""
Todo:
    1. Remove blank lines from pretrain.txt.
"""

import argparse
from copy import copy
from functools import partial
import gc
import itertools
from multiprocessing import cpu_count, Pool
import os
from pathlib import Path
import random
import sys
from typing import List, Dict
sys.path.append('..')

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from utils import logging  # noqa: E402
from utils.utils import calc_chunksize  # noqa: E402
logger = logging.set_logging(__name__)

DATA_DIR = {
    'toy': '../../data/toy',
    'FB15K-237': '../../data/FB15K-237',
    'WN18RR': '../../data/WN18RR',
    'UMLS': '../../data/UMLS',
}
DEFAULT_NUM_HOPS = 2
DEFAULT_CHUNKSIZE = None
DEFAULT_LOG_LEVEL = 'DEBUG'
PRETRAIN_FILENAME = 'pretrain/pretrain.txt'

# global variables
logger = None
_random = None
all_paths = None
entity_paths_dict = {}
entity_range_dict = {}


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
        '--num_hops',
        default=DEFAULT_NUM_HOPS,
        type=int,
        help='Depth to search the path. \
              Only paths of length <= num_hops are considered.')

    parser.add_argument(
        '--mode',
        required=True,
        type=str,
        help='Generate pretraining data for different tasks (MLM | MLM-NSP).')

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

    args = parser.parse_args()

    # Check integrity.
    if args.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
        raise ValueError(f'Invalid log level: {args.log_level}')

    if args.dataset not in DATA_DIR:
        raise ValueError(f'Invalid dataset specified: {args.dataset}')

    if args.mode not in ['MLM', 'MLM-NSP']:
        raise ValueError(f'Invalid mode specified: {args.mode}')

    return args


def _find_paths(
        entity_all_paths: Dict[str, List[str]],
        max_hop: int,
        entity_id: List[str]
        ) -> List[str]:
    """
    """
    if entity_id not in entity_all_paths:
        return None

    shorter_paths = entity_all_paths[entity_id]
    result = copy(shorter_paths)

    for _ in range(max_hop-1):
        longer_paths = []
        for first_path in shorter_paths:
            no_loopback = (first_path.split(' ')[-3], entity_id)
            second_paths = [' '.join(p.split(' ')[1:])
                            for p in entity_all_paths[first_path.split(' ')[-1]]
                            if not p.endswith(no_loopback)]
            if len(second_paths) == 0:
                continue
            cur_hop_paths = list(itertools.product([first_path], second_paths))
            longer_paths.extend([' '.join(x) for x in cur_hop_paths])
        shorter_paths = longer_paths
        result.extend(longer_paths)

    return result


def generate_mlm_pretrain_data(
        args: argparse.Namespace,
        df_train: pd.DataFrame,
        df_entities: pd.DataFrame,
        n_workers: int,
        pretrain_filepath: str = None,
        ) -> None:
    """
    """
    logger.info('Generating MLM pretrain data...')

    #
    entity_id_list = df_entities['id'].tolist()
    logger.info(f'Number of entities: {len(entity_id_list)}')

    #
    df_train_with_rev = df_train.copy().rename(columns={'head': 'tail', 'tail': 'head'})
    df_train_with_rev['relation'] = df_train_with_rev['relation'].apply(lambda x: f'!{x}')
    df_train_with_rev = pd.concat([df_train, df_train_with_rev])[['head', 'relation', 'tail']]
    df_train_with_rev['str_path'] = df_train_with_rev.apply(
        lambda row: ' '.join(row.tolist()), axis=1)

    entity_all_paths = df_train_with_rev.groupby('head')['str_path'].apply(list).to_dict()

    del df_train_with_rev
    gc.collect()

    # Save paths.
    if args.chunksize is None:
        chunksize = calc_chunksize(n_workers=n_workers, len_iterable=len(entity_id_list))
    else:
        chunksize = args.chunksize
    logger.info(f'Chunk size: {chunksize}')

    if pretrain_filepath is not None:
        Path(os.path.dirname(pretrain_filepath)).mkdir(parents=True, exist_ok=True)

        with open(pretrain_filepath, mode='w', encoding='utf-8') as file:
            logger.info(f'File open for saving pretrain data: {pretrain_filepath}')

            with Pool(n_workers) as p:
                for result in list(tqdm(p.imap(
                            partial(_find_paths, entity_all_paths, args.num_hops),
                            entity_id_list,
                            chunksize=chunksize),
                        total=len(entity_id_list))):
                    if result:
                        file.write('\n'.join(result)+'\n')

        logger.info(f'Pretrain data saved to \'{pretrain_filepath}\'.')
    else:
        pretrain_data = []
        with Pool(n_workers) as p:
            for result in list(tqdm(p.imap(
                        partial(_find_paths, entity_all_paths, args.num_hops),
                        entity_id_list,
                        chunksize=chunksize),
                    total=len(entity_id_list))):
                if result:
                    pretrain_data.extend(result)

        return pretrain_data


def generate_mlm_nsp_pretrain_data(
        args: argparse.Namespace,
        df_train: pd.DataFrame,
        df_entities: pd.DataFrame,
        n_workers: int,
        pretrain_filepath: str = None,
        sample_rate: Dict[int, float] = {1: 1.0, 2: 1.0}
        ) -> None:
    """
    """
    logger.info('Generating MLM & NSP pretrain data...')

    #
    entity_id_list = df_entities['id'].tolist()
    logger.info(f'Number of entities: {len(entity_id_list)}')

    #
    global all_paths
    all_paths = generate_mlm_pretrain_data(
        args=args,
        df_train=df_train,
        df_entities=df_entities,
        n_workers=n_workers)

    logger.info(f'Number of all paths of hop {args.num_hops}: {len(all_paths)}')

    # Do sampling and compare before and after.
    def _print_paths_stat(x: List[str]):
        paths_stat = {}
        for p in x:
            hops = p.count('r')
            if hops in paths_stat:
                paths_stat[hops] += 1
            else:
                paths_stat[hops] = 1
        logger.info(f'Number of paths for each hop: {paths_stat}')

    _print_paths_stat(all_paths)
    logger.info(f'Sampling rates for each hop: {sample_rate}')
    all_paths = [p for p in all_paths if _random.uniform(0, 1) <= sample_rate[p.count('r')]]
    _print_paths_stat(all_paths)

    #
    logger.info('Extracting entity paths and range dictionaries...')
    global entity_paths_dict
    global entity_range_dict
    for p in all_paths:
        start_entity = p.split(' ')[0]
        end_entity = p.split(' ')[-1]

        # entity_paths_dict
        if start_entity in entity_paths_dict:
            if p not in entity_paths_dict[start_entity]:
                entity_paths_dict[start_entity].append(p)
        else:
            entity_paths_dict[start_entity] = [p]

        # entity_range_dict
        if start_entity in entity_range_dict:
            if end_entity not in entity_range_dict[start_entity]:
                entity_range_dict[start_entity].append(end_entity)
        else:
            entity_range_dict[start_entity] = [end_entity]

    #
    entity_id_not_in_training_list = [
        e for e in entity_id_list
        if e not in entity_paths_dict.keys()]
    entity_id_list = [e for e in entity_id_list if e in entity_paths_dict.keys()]
    logger.info(f'Entities not in training data: {entity_id_not_in_training_list}')
    logger.info(f'Number of entities in training data: {len(entity_id_list)}')

    gc.collect()

    #
    Path(os.path.dirname(pretrain_filepath)).mkdir(parents=True, exist_ok=True)
    with open(pretrain_filepath, mode='w', encoding='utf-8') as file:
        logger.info(f'File open for saving training data: {pretrain_filepath}')

        # header
        file.write('next_sentence_label\tpath_1\tpath_2\n')

        # Use multiprocessing to speed things up.
        if args.chunksize is None:
            chunksize = calc_chunksize(n_workers=n_workers, len_iterable=len(entity_id_list))
        else:
            chunksize = args.chunksize
        logger.info(f'Chunk size: {chunksize}')

        with Pool(n_workers) as p:
            for result in list(tqdm(p.imap(
                        _make_sentence_pairs,
                        entity_id_list,
                        chunksize=chunksize),
                    total=len(entity_id_list))):
                file.write('\n'.join(result)+'\n')

    logger.info(f'File saved at \'{pretrain_filepath}\'')


def _make_sentence_pairs(entity_id: str):
    first_paths = entity_paths_dict[entity_id]
    second_paths_entity_paths_dict = {
        sp_start_entity: [p for p in entity_paths_dict[sp_start_entity]
                          if not p.endswith(entity_id)]
        for sp_start_entity in entity_range_dict[entity_id]}

    for _, val in second_paths_entity_paths_dict.items():
        _random.shuffle(val)

    result = []
    for i, fp in enumerate(first_paths):
        fp_range = fp.split(' ')[-1]

        correct_sps = second_paths_entity_paths_dict[fp_range]
        if len(correct_sps) == 0:
            continue

        randomly_chosen_positive_sp = correct_sps[i % len(correct_sps)]
        result.extend([f'1\t{fp}\t{randomly_chosen_positive_sp}'])

        randomly_chosen_negative_sp = _random.choice(all_paths)
        while randomly_chosen_negative_sp.startswith(fp_range+' '):
            randomly_chosen_negative_sp = _random.choice(all_paths)

        result.extend([f'0\t{fp}\t{randomly_chosen_negative_sp}'])

    return result


def main():
    args = parse_argument()

    # Set log.
    global logger

    logger = logging.set_logging(__name__, log_level=args.log_level)
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
    n_workers = cpu_count()
    logger.info(f'Using {n_workers} CPUs for multiprocessing.')

    # Initialize file paths.
    processed_data_dir = os.path.join(DATA_DIR[args.dataset], 'processed_data')
    logger.info(f'Processed data directory: {processed_data_dir}')

    # Load data.
    df_train = pd.read_csv(
        os.path.join(processed_data_dir, 'train.txt'),
        sep='\t',
        na_filter=False)

    df_entities = pd.read_csv(
        os.path.join(processed_data_dir, 'entities.txt'),
        sep='\t',
        na_filter=False)

    if args.mode == 'MLM':
        pretrain_filepath = os.path.join(
            processed_data_dir,
            PRETRAIN_FILENAME.replace('.txt', f'_{args.num_hops}-hops_{args.mode}.txt'))

        generate_mlm_pretrain_data(
            args=args,
            df_train=df_train,
            df_entities=df_entities,
            n_workers=n_workers,
            pretrain_filepath=pretrain_filepath)
    elif args.mode == 'MLM-NSP':
        pretrain_filepath = os.path.join(
            processed_data_dir,
            PRETRAIN_FILENAME.replace('.txt', f'_{args.num_hops}-hops_{args.mode}.txt'))

        generate_mlm_nsp_pretrain_data(
            args=args,
            df_train=df_train,
            df_entities=df_entities,
            n_workers=n_workers,
            pretrain_filepath=pretrain_filepath)


if __name__ == '__main__':
    main()
