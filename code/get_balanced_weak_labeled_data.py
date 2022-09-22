#!/usr/bin/env python
# coding: utf-8

'''
@title: Get Balanced Weak Labeled Data
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: Construct weak labeled data set from weak labeled data with similarity matrix.
'''

import os
import glob
import tqdm
import argparse
import pandas as pd


def parse_arguments(parser):
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_filepath', type=str, required=True)
    parser.add_argument('--each_class_size', type=int, default=10000)
    return parser


def main(args):
    input_dir = args.input_dir
    output_filepath = args.output_filepath

    each_class_size = args.each_class_size
    random_seed = 3407

    if not os.path.isdir(input_dir):
        raise ValueError("The specified input path does not exist.")

    # Read all files from weak labeling process
    all_tweet_files = glob.glob(input_dir + "/**/predictions.csv", recursive=True)
    print(f"There are {len(all_tweet_files)} files")
    dfs = []

    for fp in tqdm.tqdm(all_tweet_files):
        df = pd.read_csv(fp)
        df_yes = df[df["weak_label"]==1]
        df_no = df[df["weak_label"]==0]

        if df_yes.shape[0] > df_no.shape[0]:
            df_yes = df_yes.sample(n=df_no.shape[0], random_state=random_seed)
        elif df_yes.shape[0] < df_no.shape[0]:
            df_no = df_no.sample(n=df_yes.shape[0], random_state=random_seed)

        dfs.append(df_yes)
        dfs.append(df_no)

    df = pd.concat(dfs, ignore_index=True)
    print(df.shape)

    grouped_size = df.groupby("weak_label").size()
    print(grouped_size)

    if each_class_size:
        if each_class_size > min(grouped_size.values):
            raise ValueError(f"Cannot do undersampling because the `each_class_size` of {each_class_size}"
                             f" is larger than the nunmber of minority class ({min(grouped_size.values)}).")
        else:
            df_yes = df[df["weak_label"]==1].sample(n=each_class_size, random_state=random_seed)
            df_no = df[df["weak_label"]==0].sample(n=each_class_size, random_state=random_seed)
            df = pd.concat([df_yes, df_no], ignore_index=False)
    print(df.groupby("weak_label").size())

    # Write to file
    df = df.sample(frac=1.0, random_state=random_seed)
    df.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    args = parser.parse_args()
    main(args)