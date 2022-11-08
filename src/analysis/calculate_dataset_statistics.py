import math
import os
import sys
sys.path.append('..')

# import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from tqdm import tqdm  # noqa: E402

from utils import logging  # noqa: E402
logger = logging.set_logging(__name__)

datasets_dir = {
    # 'FB15K-237': '../../data/FB15K-237/processed_data',
    'WN18RR': '../../data/WN18RR/processed_data',
    # 'toy': '../../data/toy/processed_data',
}


def main():
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')

    for dataset, directory in datasets_dir.items():
        logger.info('Processing dataset %s...', dataset)

        ##########################
        # entity/relation length #
        ##########################
        # load data
        entities_filepath = os.path.join(directory, 'entities.txt')
        relations_filepath = os.path.join(directory, 'relations.txt')

        entities = pd.read_csv(
            entities_filepath, sep='\t', na_filter=False)['processed_name'].to_list()
        logger.info('Number of entities: %d', len(entities))

        relations = pd.read_csv(
            relations_filepath, sep='\t', na_filter=False)['processed_name'].to_list()
        logger.info('Number of relations: %d', len(relations))

        # encode using the tokenizer
        encoded_entities = tokenizer(entities, return_length=True)
        encoded_relations = tokenizer(relations, return_length=True)

        # find out the length
        entities_length = sorted([x-2 for x in encoded_entities['length']], reverse=True)
        relations_length = sorted([x-2 for x in encoded_relations['length']], reverse=True)

        entities_length_mean = np.mean(entities_length)
        entities_length_std = np.std(entities_length)

        relations_length_mean = np.mean(relations_length)
        relations_length_std = np.std(relations_length)

        logger.info('Length of entity tokens: %.2f±%.2f', entities_length_mean, entities_length_std)
        logger.info(f'Largest entity token length: {entities_length[0]}')
        logger.info('Length of relation tokens: %.2f±%.2f', relations_length_mean, relations_length_std)
        logger.info(f'Largest relation token length: {relations_length[0]}')

        seen = set()
        seen_add = seen.add
        entities_length_unique = [x for x in entities_length if not (x in seen or seen_add(x))]
        entities_length_unique = list(reversed(entities_length_unique))

        for i in entities_length_unique:
            coverage = np.mean([x <= i for x in entities_length])
            logger.info(f'Coverage for entity length {i}: {coverage}')

        steps = math.ceil((entities_length_unique[-1] - entities_length_unique[0]) / 10)
        bins = list(range(entities_length_unique[0], entities_length_unique[-1], steps))
        if bins[-1] != entities_length_unique[-1]:
            bins.append(entities_length_unique[-1])
        assert len(bins) == 11

        label_list = []
        coverage_list = []
        for idx, low in enumerate(bins[:-1]):
            high = bins[idx+1]

            if idx != len(bins) - 2:
                label_list.append(f'<{high}')
                local_unique_entities_length = [x for x in entities_length_unique if x >= low and x < high]
            else:
                label_list.append(f'≤{high}')
                local_unique_entities_length = [x for x in entities_length_unique if x >= low and x <= high]

            coverage = []
            for i in local_unique_entities_length:
                coverage.extend([x <= i for x in entities_length])
            coverage_list.append(np.mean(coverage))

        df_entity_stat = pd.DataFrame({'Entity Token Size': label_list, 'Coverage': coverage_list})
        df_entity_stat.plot.bar(x='Entity Token Size', y='Coverage', rot=0, width=0.7, figsize=(4, 6))

        plt.tight_layout()
        plt.show()

        break

        ############
        # pretrain #
        ############
        pretrain_filepath = os.path.join(directory, 'pretrain', 'pretrain.txt')

        hop_stat = {}
        with open(pretrain_filepath, mode='r', encoding='utf-8') as file:
            for line in tqdm(file.readlines()):
                num_hops = line.count('r')
                if num_hops not in hop_stat:
                    hop_stat[num_hops] = 1
                else:
                    hop_stat[num_hops] += 1

        logger.info(f'Hop statistics: {hop_stat}')
        logger.info('')


if __name__ == '__main__':
    main()
