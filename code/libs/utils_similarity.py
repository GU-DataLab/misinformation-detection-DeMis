#!/usr/bin/env python
# coding: utf-8

'''
@title: Utilities for similarity-related management
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: Utilities for working on similarity matrix
'''


import json
import numpy as np
import pandas as pd
from typing import Dict, Any


def most_similar(df_tweet, df_claim, doc_id, similarity_matrix, top_k, metric="Cosine Similarity"):
    print (f'Document: {df_tweet.iloc[doc_id]["text"]}')
    print (f'Cleaned Document: {df_tweet.iloc[doc_id]["text_cleaned"]}')
    print ('\n')
    print ('Similar Documents:')
    if metric == 'Cosine Similarity':
        similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]
    elif metric == 'Euclidean Distance':
        # similar_ix = np.argsort(similarity_matrix[doc_id])
        raise Exception("Euclidean Distance is not implemented yet")
    for i, ix in enumerate(similar_ix):
        if ix == doc_id:
            continue
        print('\n')
        print(f'Rank - {i+1}')
        print(f'Document: {df_claim.iloc[ix]["claim"]}')
        print(f'Cleaned Document: {df_claim.iloc[ix]["claim_cleaned"]}')
        print(f'{metric}: {similarity_matrix[doc_id][ix]}')
        print(f'Rating: {df_claim.iloc[ix]["truth_rating"]}')
        top_k -= 1
        if top_k <= 0:
            break


def save_similarity_matrix(similarities, filename, query_ids=None):
    """ Save similarity matrix into a CSV file using query IDs as keys if provided.
    """

    sim_size = similarities.shape[0]
    if query_ids and sim_size != len(query_ids):
        raise ValueError(f"Length not equal. Found {sim_size} and {len(query_ids)}.")

    sim_dict = dict()
    for i in range(sim_size):
        key = query_ids[i] if query_ids else str(i)
        sim_dict[key] = similarities[i].tolist()

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(sim_dict, f, ensure_ascii=False, indent=4)


def read_similarity_matrix(input_fp, query_ids=None) -> Dict[str, Any]:
    """ Read similarity matrix from json file   
    """
    with open(input_fp) as f:
        data = json.load(f)

    # if query_ids and sim_size != len(query_ids):
    #     raise ValueError(f"Length not equal. Found {sim_size} and {len(query_ids)}.")

    # similarities = dict()
    # for i in range(len(data)):
    #     key = query_ids[i] if query_ids else str(i)
    #     similarities.append(data[key])

    # return np.array(similarities)

    return data