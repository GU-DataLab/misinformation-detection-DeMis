#!/usr/bin/env python
# coding: utf-8

'''
@title: Utilities for HuggingFace Transformers.
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: Functions for huggingface transformers.
'''


import numpy as np
import pandas as pd


def add_special_tokens_twitter(tokenizer, tokens=["HTTPURL", "@USER"]):
    # Add special tokens for Twitter data including censored URLs and @mentions
    tokenizer.add_tokens(tokens, special_tokens=False)
    return tokenizer

def print_sequence_len_coverage_stats(tokenizer, list_texts, sequence_len, logger=None):
    # Check coverage of sequence length
    # E.g. if we set seq_len to 128, how many tweets are shorter than that.
    outputs = tokenizer(
        list_texts, padding=False, max_length=tokenizer.model_max_length, truncation=False)
    seq_lens = pd.Series(
        [sum(mask) for mask in outputs["attention_mask"]],
        dtype=int
    )
    print("##### Sequence stats #####")
    print(seq_lens.describe())
    tmp = seq_lens.tolist() + [sequence_len]
    tmp.sort(reverse=True)
    idx_of_seq_len = tmp.index(sequence_len)
    if logger:
        logger.info(f"sequence_len is {sequence_len}")
        logger.info(f"sequence_len is at index {idx_of_seq_len} from {len(seq_lens)}"
            f" (top {round(idx_of_seq_len/len(seq_lens)*100, 2)}%)")
    else:
        print(f"sequence_len is {sequence_len}")
        print(f"sequence_len is at index {idx_of_seq_len} from {len(seq_lens)}"
            f" (top {round(idx_of_seq_len/len(seq_lens)*100, 2)}%)")