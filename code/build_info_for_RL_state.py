#!/usr/bin/env python
# coding: utf-8

'''
@title: Build Info for RL State
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: Build state information for RL.
'''


from email.policy import default
import os
import csv
import argparse
import pandas as pd

from libs import utils_misc
from libs.utils_file import read_json_to_dict
from libs.utils_similarity import read_similarity_matrix


def parse_arguments(parser):
    parser.add_argument('--input_filepath', type=str, required=True)
    parser.add_argument('--output_filepath', type=str, required=True)
    parser.add_argument('--similarity_matrix_filepath', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=10)
    return parser


def main(args):
    TARGET_CLAIM_IDS = read_json_to_dict("myth_theme_to_claim_ids.json")

    # Check output directory
    filepath = args.input_filepath
    output_dir = "/".join(args.output_filepath.split("/")[:-1])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Read data
    df = pd.read_csv(filepath)
    print(df.shape)
    print(df.columns)

    # Read similarity matrix of tweet-claim
    similarities = read_similarity_matrix(args.similarity_matrix_filepath)
    print(len(similarities))

    def get_sim_highest_target_claim(list_similarities, target_claims):
        return max([list_similarities[claim_id] for claim_id in target_claims])

    top_k = args.top_k

    dict_top_claim_ids = {}
    for tweet_id in df["tweet_id"]:
        doc_id = str(tweet_id)
        top_claim_ids = utils_misc.get_indices_top_k(list(similarities[doc_id]), k=top_k)
        dict_top_claim_ids[doc_id] = top_claim_ids

    # Similarity to the top-1-th claim
    df["sim_top_1th"] = [
        max(similarities[str(tweet_id)]) for tweet_id in df["tweet_id"]]

    # Similarity to the top-k-th claim
    df["sim_top_kth"] = [
        similarities[str(tweet_id)][dict_top_claim_ids[str(tweet_id)][top_k-1]]
        for tweet_id in df["tweet_id"]]

    # Similarity to the highest matched target claim
    df["sim_highest_target_claim"] = [
        get_sim_highest_target_claim(
            similarities[str(tweet_id)], TARGET_CLAIM_IDS["weather"])
        for tweet_id in df["tweet_id"]]

    df.to_csv(args.output_filepath,\
        escapechar='\"', \
        quotechar='\"',\
        quoting=csv.QUOTE_ALL,\
        index=False)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    args = parser.parse_args()
    main(args)