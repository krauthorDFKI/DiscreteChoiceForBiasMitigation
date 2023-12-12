# Not contained in original repo: refactored for ours

import numpy as np
from scipy import sparse


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tocsr(data: np.array, num_user: int, num_item: int) -> sparse.csr_matrix:
    """Convert data to csr_matrix."""
    matrix = sparse.lil_matrix((num_user, num_item))
    for (u, i, r) in data[:, :3]:
        matrix[u, i] = r
    return sparse.csr_matrix(matrix)

def csr_to_user_dict(train_matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    train_dict = {}
    for idx, value in enumerate(train_matrix):
        train_dict[idx] = value.indices.copy().tolist()
    return train_dict

def csr_to_user_dict_neg(train_matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    train_dict_neg = {}
    unique_items = np.asarray(range(train_matrix.shape[1]))
    for idx, value in enumerate(train_matrix):
        pos_list = value.indices.copy().tolist()
        neg_items = np.setdiff1d(unique_items, pos_list)

        train_dict_neg[idx] = neg_items
    return train_dict_neg
