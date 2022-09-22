#!/usr/bin/env python
# coding: utf-8


'''
@title: Tweet Preprocessing
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: To preprocess tweets by replacing mentions and URLs.
'''


import csv
import argparse
import pandas as pd

from libs.utils_text import censor_text


def parse_arguments(parser):
    parser.add_argument('--input_tweet_filepath', type=str, required=True)
    parser.add_argument('--output_filepath', type=str, required=True)
    return parser


def main(args):
    df = pd.read_csv(args.input_tweet_filepath, dtype={"tweet_id": str})
    df["text"] = df["text"].apply(censor_text)
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