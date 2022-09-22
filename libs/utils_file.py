#!/usr/bin/env python
# coding: utf-8

'''
@title: Utilities for file-related management
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: Utilities for working on folder and files
'''


import pandas as pd
import glob
import os
import shutil
import json

from typing import Any, Dict


def read_json_to_dict(filepath: str) -> Dict[str, Any]:
    with open(filepath) as json_file:
        return json.load(json_file)

def remove_path(dirpath):
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)

def folder2dataframe(folder_path, file_type="csv", dtype={}, names=None):
    """
    Read file to a dataframe with folder path as input

    Ref: https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
    """
    all_files = glob.glob(folder_path + "/*." + file_type)
    dfs = []
    for filename in all_files:
        dfs.append(pd.read_csv(filename, index_col=None, dtype=dtype, names=names))

    df = pd.concat(dfs, axis=0, ignore_index=True)

    return df


def read_fakenewsnet_dataset_ids(filepath: str):
    """ Read FakeNewsNet tweet IDs

    Return:
        A dataframe
    """
    df = pd.read_csv(filepath, dtype=str)
    df["tweet_ids"] = df["tweet_ids"].apply(
        lambda x: [] if isinstance(x, float) else [i.strip() for i in x.split("\t")])
    return df


def read_dataverse_dataset_ids(filepath: str):
    """ Read Harvard Dataverse tweet IDs (file: corona_doc_topics.txt)

    Ref: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/9ICICY

    Return:
        A dataframe
    """
    df = pd.read_csv(filepath, sep="\t", header=None)
    df = df.rename(columns={1: "tweet_id"})  # second column contains tweet ids
    df = df[["tweet_id"]]
    return df


def read_performance_file(filename):
    """
    Example (txt):
        [0.26590215871110556]
        [0.8538591849886179]
        [0.9250814332247557]
        [0.9703703703703703]
        [0.8733333333333333]
        [0.9192982456140351]

        ### Average performance ###
        Loss: 0.26590215871110556
        Matthews: 0.8538591849886179
        Accuracy: 0.9250814332247557
        Precision: 0.9703703703703703
        Recall: 0.8733333333333333
        f1-score: 0.9192982456140351
        Confusion Matrix
        [[153   4]
        [ 19 131]]
    """

    if filename.endswith(".json"):
        with open(filename, "r") as f:
            data = json.load(f)
            
    elif filename.endswith(".txt"):
        with open(filename, "r") as f:
            data = {}
            for line in f.readlines():
                line = line.strip()
                values = line.split()
                if len(values) == 2:
                    key = str(values[0]).strip()
                    if key.endswith(":"):
                        key = key[:-1]
                        key = key.lower()
                        if key.startswith("f1"):
                            key = "f1"
                        value = float(values[1])
                        data[key] = value
    else:
        raise ValueError("File should be json or txt.")

    return data


def main():
    fakenewsnet_ids_dir = "../../data/fakenewsnet_dataset_news_articles_and_tweets/dataset_ids"
    fakenewsnet_ids_filepath = f"{fakenewsnet_ids_dir}/politifact_fake.csv"
    df = read_fakenewsnet_dataset_ids(fakenewsnet_ids_filepath)
    print(df.shape)
    print(df.head(5))

    dataverse_ids_dir = "../../data/dataverse"
    dataverse_ids_filepath = f"{dataverse_ids_dir}/corona_doc_topics.txt"
    df = read_dataverse_dataset_ids(dataverse_ids_filepath)
    print(df.shape)
    print(df.head(5))
    print(df.columns)

if __name__ == "__main__":
    main()