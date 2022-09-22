# coding: utf-8

'''
@title: Generate Sentence Embedding for DeMis
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: Generate embeddings using sentence encoder.
'''


import os
import sys
import torch
import argparse
import pandas as pd
from loguru import logger

from libs.utils_text import clean_text
from sim_sentence_embedding import SentenceEncoder


def parse_arguments(parser):
    parser.add_argument('--input_tweet_filepath', type=str, required=True)
    parser.add_argument('--output_filepath', type=str, required=True)
    return parser


def main(args):
    # Set logger level to print out
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    df_tweet = pd.read_csv(args.input_tweet_filepath, dtype={"tweet_id": str})
    logger.info(f"Total tweet: {df_tweet.shape[0]}")

    output_dir = "/".join(args.output_filepath.split("/")[:-1])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Clean tweets and claims
    df_tweet['text_cleaned'] = df_tweet["text"].apply(lambda x: clean_text(x))

    # Load model
    model_path = "sentence-transformers/bert-base-nli-stsb-mean-tokens"
    logger.info(f"Load Model: {model_path}")
    encoder = SentenceEncoder(model_path)

    # Generate embeddings
    sentence_embeddings = encoder.get_embeddings(
        df_tweet["text_cleaned"].tolist(), pooling="mean", verbose=True)
    assert len(sentence_embeddings) == df_tweet.shape[0]

    # Build dict and save
    tweet_ids_to_sentence_embeddings = {}
    for i, tweet_id in enumerate(df_tweet["tweet_id"]):
        tweet_ids_to_sentence_embeddings[tweet_id] = sentence_embeddings[i]
    torch.save(tweet_ids_to_sentence_embeddings, args.output_filepath)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    args = parser.parse_args()
    main(args)