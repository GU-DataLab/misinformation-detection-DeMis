#!/usr/bin/env python
# coding: utf-8

'''
@title: Hit@k
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: Calculate accuracy of tweet-claim pairs using Hits@k.
'''

import sys
import glob
import tqdm
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

sys.path.append("../.")
from libs import utils_misc
from libs.utils_similarity import read_similarity_matrix


def found_matched_claim(correct_claim_ids, sorted_potential_claim_ids):
    if len(set(correct_claim_ids).intersection(set(sorted_potential_claim_ids))) > 0:
        return True
    else:
        return False


def main():
    # Argument setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filepath_tweet", type=str, required=True)
    parser.add_argument("--input_filepath_claim", type=str, required=True)
    parser.add_argument("--similarity_matrix_folder", type=str, required=True)
    parser.add_argument("--k_values", type=str, required=False, default="1,3,5,10",
                        help='Can take multiple k\'s, separated by ",".')
    parser.add_argument("--correct_claim_ids", type=str, required=True,
                        help='Can take multiple IDs, separated by ",".')
    parser.add_argument("--balance_testing", required=False, default=False, action='store_true')
    parser.add_argument("--random_seed", type=int, required=False, default=1)
    args = parser.parse_args()

    # Read tweets
    df_tweet = pd.read_csv(args.input_filepath_tweet, dtype={"tweet_id": str})
    if args.balance_testing:
        df_tweet_yes = df_tweet[df_tweet["is_myth"]=="yes"]
        df_tweet_no = df_tweet[df_tweet["is_myth"]!="yes"]
        if df_tweet_yes.shape[0] < df_tweet_no.shape[0]:
            # Undersampling not-myth tweets
            df_tweet_no = df_tweet_no.sample(
                n=df_tweet_yes.shape[0], random_state=args.random_seed)
        df_tweet = pd.concat([df_tweet_yes, df_tweet_no])
    print("Total tweet:", df_tweet.shape[0])
    print(df_tweet.groupby("is_myth").size())

    # Read claims
    df_claim = pd.read_csv(args.input_filepath_claim)
    print("Total claim:", df_claim.shape[0])

    # Manually check and found these are the correctly matched claims.
    correct_claim_ids = [int(s.strip()) for s in args.correct_claim_ids.split(",")]

    # Specify k for Hits@k
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    # Store accuracies
    accuracy_at_k = {k: [] for k in k_values}
    precision_at_k = {k: [] for k in k_values}
    recall_at_k = {k: [] for k in k_values}
    f1_at_k = {k: [] for k in k_values}
    model_names = []

    # Read similarity matrix
    all_files = glob.glob(args.similarity_matrix_folder + "/**/*.json", recursive=True)
    for matrix_file_path in tqdm.tqdm(all_files):
        similarities = read_similarity_matrix(matrix_file_path)
        model_names.append(matrix_file_path.split("/")[-1])

        # Get top K potential claims of each tweet
        list_top_claim_ids = []
        for idx in range(df_tweet.shape[0]):
            doc_id = str(idx)
            if doc_id not in similarities:
                doc_id = df_tweet.iloc[idx]["tweet_id"]  # Extract tweet IDs

            top_claim_ids = utils_misc.get_indices_top_k(list(similarities[doc_id]), k=max(k_values))
            list_top_claim_ids.append(top_claim_ids)
        df_tweet["top_claim_ids"] = list_top_claim_ids

        # Calculate Hits@k performance
        true_labels = [1 if myth == "yes" else 0 for myth in df_tweet["is_myth"]]
        for k in k_values:
            predictions = []
            for top_claim_ids in df_tweet["top_claim_ids"]:
                if found_matched_claim(correct_claim_ids, top_claim_ids[:k]):
                    predictions.append(1)
                else:
                    predictions.append(0)

            acc = accuracy_score(true_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="binary")
            accuracy_at_k[k].append(acc)
            precision_at_k[k].append(precision)
            recall_at_k[k].append(recall)
            f1_at_k[k].append(f1)


    df_score = pd.DataFrame()
    df_score["model_name"] = model_names
    for k in k_values:
        df_score["Acc Hits@{}".format(k)] = accuracy_at_k[k]
        df_score["Pr Hits@{}".format(k)] = precision_at_k[k]
        df_score["Re Hits@{}".format(k)] = recall_at_k[k]
        df_score["F1 Hits@{}".format(k)] = f1_at_k[k]

    df_score = df_score.sort_values(
        by=[c for c in df_score.columns if c.startswith("Re Hits@")],
        ignore_index=True,
        ascending=False)
    print(df_score)
    

if __name__ == "__main__":
    main()