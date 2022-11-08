import argparse
from glob import glob
import os
import sys
sys.path.append('..')

import pandas as pd  # noqa: E402

from utils.utils import read_data  # noqa: E402

OUTPUT_DIR = '../../output'
VAL_RESULTS_FOLDER = 'val_results'


def parse_argument() -> argparse.Namespace:
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Find best model from validation results.')

    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset (WN18RR | FB15K-237).')

    parser.add_argument(
        '--save_results',
        type=str,
        default=None,
        help='Filepath to save the collated results dataframe.')

    parser.add_argument(
        '--exclude_filter',
        type=str,
        default=None,
        help='Specify any string that cannot be in the filepath.')

    parser.add_argument(
        '--include_filter',
        type=str,
        default=None,
        help='Specify any string that must be in the filepath.')

    args = parser.parse_args()

    # Check integrity.
    if args.dataset not in ['WN18RR', 'FB15K-237']:
        raise ValueError(f'Invalid dataset: {args.dataset}')

    return args


def main():
    args = parse_argument()

    # find validation folders for the specified dataset
    val_result_folders = glob(
        os.path.join(OUTPUT_DIR, '**', VAL_RESULTS_FOLDER),
        recursive=True)
    val_result_folders = [x for x in val_result_folders if args.dataset in x]

    if args.exclude_filter:
        val_result_folders = [x for x in val_result_folders if args.exclude_filter not in x]

    if args.include_filter:
        val_result_folders = [x for x in val_result_folders if args.include_filter in x]

    print(f'{len(val_result_folders)} validation result folders found for {args.dataset}')

    results_dict = {
        'er_type_mode': [],
        'pretrain_hops': [],
        'pretrain_epochs': [],
        'pretrain_checkpoint': [],
        'num_finetune_negatives': [],
        'finetune_corrupt_mode': [],
        'finetune_forward_andor_reverse': [],
        'finetune_train_epochs': [],
        'finetune_lr': [],
        'finetune_wd': [],
        'finetune_bs': [],
        'checkpoint': [],
        # 'alpha': [],
        'mr': [],
        'mrr': [],
        'hits_1': [],
        'hits_3': [],
        'hits_10': [],
    }
    for folder in val_result_folders:
        if not folder.endswith(VAL_RESULTS_FOLDER):
            raise ValueError(f'Invalid folder: {folder}')
        print(f'Loading results from folder: {folder}')

        folder_split = folder.split('/')
        # er_type_mode = folder_split[4].split('_')[7]
        er_type_mode = 'n/a'
        # pretrain_hops = folder_split[4].split('_')[8].split('-')[-1]
        # pretrain_epochs = folder_split[4].split('_')[9].split('-')[-1]
        # pretrain_checkpoint = folder_split[4].split('_')[10] if 'checkpoint' in folder_split[4] else 'final'
        pretrain_hops = 'n/a'
        pretrain_epochs = 'n/a'
        pretrain_checkpoint = 'n/a'
        num_finetune_negatives = folder_split[6].split('_')[1].split('-')[0]
        finetune_corrupt_mode = folder_split[6].split('_')[2]
        finetune_forward_andor_reverse = folder_split[6].split('_')[-1]
        finetune_train_epochs = folder_split[7].split('_')[0].split('-')[-1]
        finetune_lr = folder_split[7].split('_')[1][3:]
        finetune_wd = folder_split[7].split('_')[2].split('-')[-1]
        finetune_bs = folder_split[7].split('_')[3].split('-')[-1]

        all_files = glob(os.path.join(folder, 'results*.txt'))
        print(f'Found {len(all_files)} files')

        for file in all_files:
            finetune_checkpoint = file.split('/')[-1][0:-4].split('_')[-1]
            # finetune_alpha = file.split('/')[-1][0:-4].split('_')[-1].split('-')[-1]
            # print(f'Loading results from checkpoint: {finetune_checkpoint}, alpha: {finetune_alpha}')

            result = read_data(file, skip_header=False)

            results_dict['er_type_mode'].append(er_type_mode)
            results_dict['pretrain_hops'].append(pretrain_hops)
            results_dict['pretrain_epochs'].append(pretrain_epochs)
            results_dict['pretrain_checkpoint'].append(pretrain_checkpoint)
            results_dict['num_finetune_negatives'].append(num_finetune_negatives)
            results_dict['finetune_corrupt_mode'].append(finetune_corrupt_mode)
            results_dict['finetune_forward_andor_reverse'].append(finetune_forward_andor_reverse)
            results_dict['finetune_train_epochs'].append(finetune_train_epochs)
            results_dict['finetune_lr'].append(finetune_lr)
            results_dict['finetune_wd'].append(finetune_wd)
            results_dict['finetune_bs'].append(finetune_bs)
            results_dict['checkpoint'].append(finetune_checkpoint)
            # results_dict['alpha'].append(finetune_alpha)
            results_dict['mr'].append(float(result[2].split(' ')[-1]))
            results_dict['mrr'].append(float(result[5].split(' ')[-1]))
            results_dict['hits_1'].append(float(result[8].split(' ')[-1]))
            results_dict['hits_3'].append(float(result[11].split(' ')[-1]))
            results_dict['hits_10'].append(float(result[17].split(' ')[-1]))

    df_results = pd.DataFrame.from_dict(results_dict)

    df_results['mr_rank'] = df_results['mr'].rank(method='min', ascending=True)
    df_results['mrr_rank'] = df_results['mrr'].rank(method='min', ascending=False)
    df_results['hits1_rank'] = df_results['hits_1'].rank(method='min', ascending=False)
    df_results['hits3_rank'] = df_results['hits_3'].rank(method='min', ascending=False)
    df_results['hits10_rank'] = df_results['hits_10'].rank(method='min', ascending=False)

    df_results['rank_sum'] = df_results.apply(
        lambda row: row['mr_rank'] + row['mrr_rank'] + row['hits1_rank'] +
        row['hits3_rank'] + row['hits10_rank'],
        axis=1)

    # df_results['rank_sum'] = df_results.apply(
    #     lambda row: row['mr_rank'] + row['hits10_rank'],
    #     axis=1)

    df_results['final_rank'] = df_results['rank_sum'].rank(method='min', ascending=True)
    df_results.sort_values(by='final_rank', inplace=True)

    if args.save_results:
        df_results.to_csv(args.save_results, sep='\t', index=False)


if __name__ == '__main__':
    main()
