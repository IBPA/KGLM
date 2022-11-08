from logging import INFO
from typing import List, Dict, Optional

import numpy as np
from transformers.utils import logging

logger = logging.get_logger(__name__)
logger.setLevel(INFO)


def _decode_path(
        path_items_list: List[str],
        entity_lookup: Dict[str, str],
        relation_lookup: Dict[str, str],
        entity_relation_type_mode: int,
        ) -> Dict[str, str]:
    def _translate(x):
        if 'e' in x:
            return entity_lookup[x]
        else:
            return relation_lookup[x.replace('!', '')]

    def _entity_relation_type_id(x):
        # Assign entity/relation type ids.
        # '0': entity
        # '1': forward relation
        # '2': inverse relation
        #
        # if entity_relation_type_mode is not 0,
        # there will be additional type ids.
        #
        # Example
        # x: born in
        # assuming 'born in' has type id 1
        # return:  1,1,1,1,1,1,1
        if 'e' in x:
            entity_length = len(entity_lookup[x])
            if entity_relation_type_mode in [0, 1]:
                return '0,'*(entity_length-1) + '0'
        elif '!r' in x:
            relation_length = len(relation_lookup[x.replace('!', '')])
            if entity_relation_type_mode == 0:
                return '2,'*(relation_length-1) + '2'
            elif entity_relation_type_mode == 1:
                relation_id = int(x.replace('!r', ''))
                return f'{relation_id*2 + 2},'*(relation_length-1) + f'{relation_id*2 + 2}'
        else:
            relation_length = len(relation_lookup[x])
            if entity_relation_type_mode == 0:
                return '1,'*(relation_length-1) + '1'
            elif entity_relation_type_mode == 1:
                relation_id = int(x.replace('r', ''))
                return f'{relation_id*2 + 1},'*(relation_length-1) + f'{relation_id*2 + 1}'

    translated = []
    entity_relation_type_ids = []
    for x in path_items_list:
        translated.append(_translate(x))
        entity_relation_type_ids.append(_entity_relation_type_id(x))

    return (translated, entity_relation_type_ids)


def _decode_function(example, **kwargs):
    entity_lookup = kwargs.pop('entity_lookup')
    relation_lookup = kwargs.pop('relation_lookup')
    entity_relation_type_mode = kwargs.pop('entity_relation_type_mode')
    model_name = kwargs.pop('model_name')
    is_finetune = kwargs.pop('is_finetune')

    if is_finetune or 'roberta-' in model_name:
        path_decoded = []
        path_entity_relation_type_ids = []
        for p in example['path']:
            decoded, entity_relation_type_ids = _decode_path(
                p.split(' '), entity_lookup, relation_lookup, entity_relation_type_mode)
            path_decoded.append(' '.join(decoded))
            path_entity_relation_type_ids.append(', ,'.join(entity_relation_type_ids))

        return {'path_decoded': path_decoded,
                'path_entity_relation_type_ids': path_entity_relation_type_ids}
    else:
        path_1_decoded = []
        path_1_entity_relation_type_ids = []
        for p in example['path_1']:
            decoded, entity_relation_type_ids = _decode_path(
                p.split(' '), entity_lookup, relation_lookup, entity_relation_type_mode)
            path_1_decoded.append(' '.join(decoded))
            path_1_entity_relation_type_ids.append(', ,'.join(entity_relation_type_ids))

        path_2_decoded = []
        path_2_entity_relation_type_ids = []
        for p in example['path_2']:
            decoded, entity_relation_type_ids = _decode_path(
                p.split(' '), entity_lookup, relation_lookup, entity_relation_type_mode)
            path_2_decoded.append(' '.join(decoded))
            path_2_entity_relation_type_ids.append(', ,'.join(entity_relation_type_ids))

        return {'path_1_decoded': path_1_decoded,
                'path_1_entity_relation_type_ids': path_1_entity_relation_type_ids,
                'path_2_decoded': path_2_decoded,
                'path_2_entity_relation_type_ids': path_2_entity_relation_type_ids}


def _tokenize_function(examples, **kwargs):
    tokenizer = kwargs.pop('tokenizer')
    model_name = kwargs.pop('model_name')
    is_finetune = kwargs.pop('is_finetune')

    if is_finetune or 'roberta-' in model_name:
        return tokenizer(
            examples['path_decoded'],
            padding=False,
            truncation=False,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            return_length=True,
        )
    else:
        return tokenizer(
            examples['path_1_decoded'],
            examples['path_2_decoded'],
            padding=False,
            truncation=False,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            return_length=True,
        )


def _assign_entity_relation_type_ids_function(examples, **kwargs):
    model_name = kwargs.pop('model_name')
    is_finetune = kwargs.pop('is_finetune')

    entity_relation_type_ids = []
    if is_finetune or 'roberta-' in model_name:
        zipped = zip(
            examples['offset_mapping'],
            examples['path_entity_relation_type_ids'],
            examples['special_tokens_mask'],
        )

        for (offset_mapping,
             path_entity_relation_type_ids,
             special_tokens_mask) in zipped:

            sub_result = []
            for idx, om in enumerate(offset_mapping):
                if special_tokens_mask[idx]:
                    sub_result.append(0)
                else:
                    sub_result.append(int(path_entity_relation_type_ids.split(',')[om[0]]))

            entity_relation_type_ids.append(sub_result)
    else:
        zipped = zip(
            examples['offset_mapping'],
            examples['path_1_entity_relation_type_ids'],
            examples['path_2_entity_relation_type_ids'],
            examples['special_tokens_mask'],
            examples['token_type_ids'],
        )

        for (offset_mapping,
             path_1_entity_relation_type_ids,
             path_2_entity_relation_type_ids,
             special_tokens_mask,
             token_type_ids) in zipped:

            sub_result = []
            for idx, om in enumerate(offset_mapping):
                if special_tokens_mask[idx]:
                    sub_result.append(0)
                elif token_type_ids[idx] == 0:
                    sub_result.append(int(path_1_entity_relation_type_ids.split(',')[om[0]]))
                else:
                    sub_result.append(int(path_2_entity_relation_type_ids.split(',')[om[0]]))

            entity_relation_type_ids.append(sub_result)

    return {'entity_relation_type_ids': entity_relation_type_ids}


def _truncate_function(examples, **kwargs):
    seq_truncate_length = kwargs.pop('seq_truncate_length')
    model_name = kwargs.pop('model_name')

    if 'roberta-' in model_name:
        zipped = zip(
                examples['attention_mask'],
                examples['entity_relation_type_ids'],
                examples['input_ids'],
                examples['special_tokens_mask'],
            )

        updated_attention_mask = []
        updated_entity_relation_type_ids = []
        updated_input_ids = []
        updated_special_tokens_mask = []

        for (attention_mask,
             entity_relation_type_ids,
             input_ids,
             special_tokens_mask) in zipped:
            # skip if no need to truncate
            path_seq_length = len(entity_relation_type_ids)
            if path_seq_length <= seq_truncate_length:
                updated_attention_mask.append(attention_mask)
                updated_entity_relation_type_ids.append(entity_relation_type_ids)
                updated_input_ids.append(input_ids)
                updated_special_tokens_mask.append(special_tokens_mask)
                continue

            prev_er_type_id = entity_relation_type_ids[special_tokens_mask.index(0)]
            entity_relation_idx = []
            local_idx = []
            for idx, er_type_id in enumerate(entity_relation_type_ids):
                if special_tokens_mask[idx] == 1:
                    continue

                if prev_er_type_id != er_type_id:
                    entity_relation_idx.append(local_idx)
                    local_idx = []
                local_idx.append(idx)

                prev_er_type_id = er_type_id
            entity_relation_idx.append(local_idx)

            # don't remove tokens from the relation
            assert len(entity_relation_idx) == 3
            del entity_relation_idx[1]

            idx_to_remove = []
            for _ in range(path_seq_length - seq_truncate_length):
                argmax = np.argmax([len(x) for x in entity_relation_idx])
                idx_to_remove.append(entity_relation_idx[argmax].pop())

            updated_attention_mask.append(np.delete(attention_mask, idx_to_remove).tolist())
            updated_entity_relation_type_ids.append(
                np.delete(entity_relation_type_ids, idx_to_remove).tolist())
            updated_input_ids.append(np.delete(input_ids, idx_to_remove).tolist())
            updated_special_tokens_mask.append(np.delete(special_tokens_mask, idx_to_remove).tolist())

        return {
            'attention_mask': updated_attention_mask,
            'entity_relation_type_ids': updated_entity_relation_type_ids,
            'input_ids': updated_input_ids,
            'special_tokens_mask': updated_special_tokens_mask,
        }
    else:
        zipped = zip(
                examples['attention_mask'],
                examples['entity_relation_type_ids'],
                examples['input_ids'],
                examples['special_tokens_mask'],
                examples['token_type_ids'],
            )

        updated_attention_mask = []
        updated_entity_relation_type_ids = []
        updated_input_ids = []
        updated_special_tokens_mask = []
        updated_token_type_ids = []

        for (attention_mask,
             entity_relation_type_ids,
             input_ids,
             special_tokens_mask,
             token_type_ids) in zipped:
            # skip if no need to truncate
            path_seq_length = len(entity_relation_type_ids)
            if path_seq_length <= seq_truncate_length:
                updated_attention_mask.append(attention_mask)
                updated_entity_relation_type_ids.append(entity_relation_type_ids)
                updated_input_ids.append(input_ids)
                updated_special_tokens_mask.append(special_tokens_mask)
                updated_token_type_ids.append(token_type_ids)
                continue

            prev_er_type_id = entity_relation_type_ids[special_tokens_mask.index(0)]
            entity_relation_idx = []
            local_idx = []
            for idx, er_type_id in enumerate(entity_relation_type_ids):
                if special_tokens_mask[idx] == 1:
                    continue

                if prev_er_type_id != er_type_id:
                    entity_relation_idx.append(local_idx)
                    local_idx = []
                local_idx.append(idx)

                prev_er_type_id = er_type_id
            entity_relation_idx.append(local_idx)

            idx_to_remove = []
            for _ in range(path_seq_length - seq_truncate_length):
                argmax = np.argmax([len(x) for x in entity_relation_idx])
                idx_to_remove.append(entity_relation_idx[argmax].pop())

            updated_attention_mask.append(np.delete(attention_mask, idx_to_remove).tolist())
            updated_entity_relation_type_ids.append(
                np.delete(entity_relation_type_ids, idx_to_remove).tolist())
            updated_input_ids.append(np.delete(input_ids, idx_to_remove).tolist())
            updated_special_tokens_mask.append(np.delete(special_tokens_mask, idx_to_remove).tolist())
            updated_token_type_ids.append(np.delete(token_type_ids, idx_to_remove).tolist())

        return {
            'attention_mask': updated_attention_mask,
            'entity_relation_type_ids': updated_entity_relation_type_ids,
            'input_ids': updated_input_ids,
            'special_tokens_mask': updated_special_tokens_mask,
            'token_type_ids': updated_token_type_ids,
        }


def preprocess_data(
        dataset,
        num_proc,
        load_from_cache_file,
        entity_lookup,
        relation_lookup,
        entity_relation_type_mode,
        tokenizer,
        max_seq_length,
        model_max_length,
        model_name,
        is_finetune):
    # decode
    logger.info('Decoding dataset...')
    processed_dataset = dataset.map(
        _decode_function,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        fn_kwargs={
            'entity_lookup': entity_lookup,
            'relation_lookup': relation_lookup,
            'entity_relation_type_mode': 0 if entity_relation_type_mode is None else entity_relation_type_mode,
            'model_name': model_name,
            'is_finetune': is_finetune,
        },
    )
    logger.info(f'Dataset after decoding: {processed_dataset}')
    logger.info(f'Sample processed dataset: {processed_dataset[:5]}')

    # tokenize
    logger.info('Tokenizing dataset...')
    processed_dataset = processed_dataset.map(
        _tokenize_function,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        fn_kwargs={
            'tokenizer': tokenizer,
            'model_name': model_name,
            'is_finetune': is_finetune,
        },
    )
    logger.info(f'Dataset after tokenization: {processed_dataset}')
    logger.info(f'Sample processed dataset: {processed_dataset[:5]}')

    sequence_length = processed_dataset['length']
    logger.info(f'Maximum model sequence length: {model_max_length}')
    logger.info(f'Maximum sequence length: {max(sequence_length)}')
    logger.info(f'Minimum sequence length: {min(sequence_length)}')
    logger.info(f'Sequence length mean: {np.mean(sequence_length)}')
    logger.info(f'Sequence length std: {np.std(sequence_length)}')

    if max_seq_length < 1:
        logger.info(f'User specified maximum sequence length percentile: {max_seq_length}')
    else:
        logger.info(f'User specified maximum sequence length cutoff: {max_seq_length}')

    # calculate optimal sequence length
    seen = set()
    seen_add = seen.add
    sequence_length_unique = [x for x in sequence_length if not (x in seen or seen_add(x))]
    sequence_length_unique = list(reversed(sorted(sequence_length_unique)))

    for i in sequence_length_unique:
        coverage = np.mean(np.array(sequence_length) <= i)
        logger.info(f'Coverage for sequence length {i}: {coverage}')

        if (coverage >= max_seq_length) and (max_seq_length <= 1):
            seq_truncate_length = i

    if max_seq_length <= 1:
        logger.info(
            f'Selected sequence truncate length for {max_seq_length} '
            f'percentile: {seq_truncate_length}')
    else:
        seq_truncate_length = int(max_seq_length)
        logger.info(
            f'Setting sequence truncate length to user specified value: {seq_truncate_length}')

    if seq_truncate_length > model_max_length:
        logger.warning(
            f'Selected sequence truncate length is larger than model max length '
            f'({model_max_length}). Forcing sequence truncate length to model max length.')
        seq_truncate_length = model_max_length

    # assign entity/relation type IDs
    logger.info('Assigning entity/relation type IDs to the dataset...')

    if is_finetune or 'roberta-' in model_name:
        remove_columns = [
            'length',
            'path',
            'path_decoded',
            'path_entity_relation_type_ids',
            'offset_mapping']
    else:
        remove_columns = [
            'length',
            'path_1',
            'path_1_decoded',
            'path_1_entity_relation_type_ids',
            'path_2',
            'path_2_decoded',
            'path_2_entity_relation_type_ids',
            'offset_mapping']

    processed_dataset = processed_dataset.map(
        _assign_entity_relation_type_ids_function,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        fn_kwargs={'model_name': model_name, 'is_finetune': is_finetune},
        remove_columns=remove_columns,
    )

    logger.info(f'Dataset after assigning entity/relation type IDs: {processed_dataset}')
    logger.info(f'Sample processed dataset: {processed_dataset[:5]}')

    # truncate
    logger.info('Truncating paths...')
    processed_dataset = processed_dataset.map(
        _truncate_function,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        fn_kwargs={
            'seq_truncate_length': seq_truncate_length,
            'model_name': model_name
        },
    )
    logger.info(f'Dataset after truncation: {processed_dataset}')
    logger.info(f'Sample processed dataset: {processed_dataset[:5]}')

    if entity_relation_type_mode is None:
        logger.info('Removing column entity_relation_type_ids')
        processed_dataset = processed_dataset.remove_columns('entity_relation_type_ids')
        logger.info(f'Dataset after removing entity_relation_type_ids: {processed_dataset}')
        logger.info(f'Sample processed dataset: {processed_dataset[:5]}')
    else:
        logger.info('Keeping column entity_relation_type_ids')

    return processed_dataset
