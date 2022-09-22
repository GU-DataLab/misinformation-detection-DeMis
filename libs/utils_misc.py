#!/usr/bin/env python
# coding: utf-8

'''
@title: Utilities for miscellaneous.
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: Functions for doing stuff.
'''


import random
import torch
import numpy as np


def get_indices_top_k(values, k):
    """ Get indices of top K values in a list."""
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:k]


def batch(arr, batch_size):
    """ Generate batches for batch processing."""
    L = len(arr)
    for idx in range(0, L, batch_size):
        yield arr[idx:min(idx + batch_size, L)]


def get_common_values_from_list_of_lists(list_of_lists):
    if len(list_of_lists) < 2:
        return None

    S = set(list_of_lists[0])
    for l in list_of_lists[1:]:
        S = S.intersection(set(l))

    return list(S)


def set_seed(k):
    random.seed(k)
    np.random.seed(k)
    torch.manual_seed(k)


def main():
    arr = [5,3,1,4,10]
    print(get_indices_top_k(arr, k=3))  # [4, 0, 3]

    for b in batch(arr, batch_size=2):
        print(b)

    list_of_arr = [
        [1,2,3,4],
        [2,3,4,5],
        [3,4,5,6]
    ]
    print(get_common_values_from_list_of_lists(list_of_arr))  # [3, 4]


if __name__ == "__main__":
    main()