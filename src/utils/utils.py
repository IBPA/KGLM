import pickle
from typing import Any, List


def save_pkl(obj: Any, save_to: str) -> None:
    """
    Pickle the given object.

    Args:
        obj: Object to pickle.
        save_to: Filepath to pickle the object to.
    """
    with open(save_to, 'wb') as fid:
        pickle.dump(obj, fid)


def load_pkl(load_from: str) -> Any:
    """
    Load the pickled object.

    Args:
        save_to: Filepath to pickle the object to.

    Returns:
        Loaded object.
    """
    with open(load_from, 'rb') as fid:
        obj = pickle.load(fid)

    return obj


def calc_chunksize(n_workers: int, len_iterable: int, factor: int = 4):
    """
    Calculate chunksize argument for Pool-methods.
    Taken from @Darkonaut https://tinyurl.com/2d4fmv4h.

    Resembles source-code within `multiprocessing.pool.Pool._map_async`.
    """
    chunksize, extra = divmod(len_iterable, n_workers * factor)
    if extra:
        chunksize += 1
    return chunksize


def read_data(
        filepath: str,
        skip_header: bool = True,
        num_header_lines: int = 1,
        ) -> List[str]:
    lines = []
    with open(filepath, mode='r', encoding='utf-8') as f:
        # skip header
        if skip_header:
            for _ in range(num_header_lines):
                next(f)

        for row in f:
            if row == '\n':
                continue
            lines.append(row[:-1])
    return lines
