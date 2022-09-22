#!/usr/bin/env python
# coding: utf-8

'''
@title: Weak Classifier
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: Weak classifier to label new samples
'''


import os
import sys
import argparse
import tqdm
import pandas as pd

from loguru import logger
from libs import utils_misc
from libs.utils_text import clean_text
from libs.utils_similarity import read_similarity_matrix
from libs.utils_similarity import save_similarity_matrix
from libs.performance import accuracy_precision_recall_fscore_support
from libs.hits_at_k import found_matched_claim
from sim_sentence_embedding import SentenceEncoder
from sim_sentence_embedding import get_similarity_matrix


class WeakClassifier():
    def __init__(
        self,
        correct_claim_ids: list,
        sentence_similarity_model_path: str = None,
        top_k: int = 10):

        # Store object variables
        self.correct_claim_ids = correct_claim_ids
        self.top_k = top_k
        if sentence_similarity_model_path:
            self.encoder = SentenceEncoder(sentence_similarity_model_path)
        else:
            self.encoder = None

    def predict(self, sorted_potential_claim_ids):
        if found_matched_claim(self.correct_claim_ids, sorted_potential_claim_ids[:self.top_k]):
            return 1  # predict is_myth == yes
        else:
            return 0  # predict is_myth == no

    def compute_similarity_matrix(self, df_tweet, df_claim, clean="", pooling="mean", verbose=False):
        if self.encoder is None:
            raise ValueError(
                "To use `compute_similarity_matrix`, please specify "
                "`sentence_similarity_model_path` when create sentence "
                "encoder from SentenceEncoder class")
        # Clean tweets and claims
        if len(clean) > 0:
            tqdm.tqdm.pandas()
            print("Cleaning tweets:")
            df_tweet['text_cleaned'] = df_tweet["text"].progress_apply(lambda x: clean_text(x))
            print("Cleaning claims:")
            df_claim["claim_cleaned"] = df_claim["claim"].progress_apply(lambda x: clean_text(x))
        if verbose:
            print("Get tweet embeddings:")

        sentence_embeddings_A = self.encoder.get_embeddings(
            df_tweet[f"text{clean}"].tolist(), pooling=pooling, verbose=verbose)
        if verbose:
            print("Get claim embeddings:")
        sentence_embeddings_B = self.encoder.get_embeddings(
            df_claim[f"claim{clean}"].tolist(), pooling=pooling, verbose=verbose)
        return get_similarity_matrix(sentence_embeddings_A, sentence_embeddings_B)


def main():
    # Argument setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filepath_tweet", type=str, required=True)
    parser.add_argument("--input_filepath_claim", type=str, required=True)
    parser.add_argument("--correct_claim_ids", type=str, required=True,
                        help='Can take multiple IDs, separated by ",".')
    parser.add_argument("--similarity_model", type=str, required=False, default=None)
    parser.add_argument("--similarity_matrix_filepath", type=str, required=False, default=None)
    parser.add_argument("--top_k", type=int, required=False, default=10)
    parser.add_argument("--save_dir", type=str, required=False, default=None)
    parser.add_argument("--do_eval", required=False, default=False, action="store_true")
    parser.add_argument("--balance_testing", required=False, default=False, action="store_true")
    parser.add_argument("--random_seed", type=int, required=False, default=3407)
    parser.add_argument("--pooling", type=str, required=False, default="mean")
    parser.add_argument("--clean", type=str, required=False, default="_cleaned")
    parser.add_argument("--sample_size", type=int, required=False, default=None)
    args = parser.parse_args()

    # Set logger level to print out
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if not os.path.isdir(args.save_dir):
        path = os.path.join(args.save_dir)
        os.makedirs(path, exist_ok=True)

    # Read tweets
    df_tweet = pd.read_csv(
        args.input_filepath_tweet, encoding="utf-8", dtype={"tweet_id": str, "id_str": str})
    # Change tweet ID column name
    if "id_str" in df_tweet.columns and "tweet_id" not in df_tweet.columns:
        df_tweet = df_tweet.rename(columns={"id_str": "tweet_id"})
    total_tweet_size = df_tweet.shape[0]
    logger.info("Total tweet: {}".format(total_tweet_size))

    if args.sample_size:
        df_tweet = df_tweet.sample(n=args.sample_size, random_state=args.random_seed)
        sampled_tweet_size = df_tweet.shape[0]
        logger.info("Sampled tweet: {}".format(sampled_tweet_size))

    if "is_myth" in df_tweet.columns:
        distribution_is_myth = df_tweet.groupby("is_myth").size()
        print(distribution_is_myth)

    # Read claims
    df_claim = pd.read_csv(args.input_filepath_claim)
    logger.info("Total claim: {}".format(df_claim.shape[0]))

    # Manually check and found these are the correctly matched claims.
    correct_claim_ids = [int(s.strip()) for s in args.correct_claim_ids.split(",")]

    # Specify k for Hits@k
    logger.info("k value: {}".format(args.top_k))

    # Declare the weak classifier
    weak_classifier = WeakClassifier(
        correct_claim_ids=correct_claim_ids,
        sentence_similarity_model_path=args.similarity_model,
        top_k=args.top_k)

    # Calculate similarity matrix using the specific model
    output_fp = None
    if args.similarity_model:
        sim_scores = weak_classifier.compute_similarity_matrix(
            df_tweet, df_claim, clean=args.clean, pooling=args.pooling, verbose=True)

        # Save similarity matrix
        clean = args.clean.replace("_", "-")
        model_name = args.similarity_model.replace("/", "-")
        output_dir = args.save_dir if args.save_dir else "."
        output_fp = f"{output_dir}/sim-{model_name}-{args.pooling}{args.clean}.json"
        save_similarity_matrix(
            sim_scores, filename=output_fp, query_ids=list(df_tweet["tweet_id"]))
        # Read similarity matrix in its format
        similarities = read_similarity_matrix(output_fp)


    # Read similarity matrix (tweet size, claim size)
    elif args.similarity_matrix_filepath:
        similarities = read_similarity_matrix(args.similarity_matrix_filepath)
    else:
        raise ValueError("Please specify either `similarity_model` or "
                         "`similarity_matrix_filepath`")

    # Get top K potential claims of each tweet
    list_top_claim_ids = []
    for idx in range(df_tweet.shape[0]):
        doc_id = str(idx)
        if doc_id not in similarities:
            doc_id = df_tweet.iloc[idx]["tweet_id"]  # Extract tweet IDs

        top_claim_ids = utils_misc.get_indices_top_k(list(similarities[doc_id]), k=args.top_k)
        list_top_claim_ids.append(top_claim_ids)
    df_tweet["top_claim_ids"] = list_top_claim_ids

    # Predict weak labels
    df_tweet["weak_label"] = [
        weak_classifier.predict(sorted_potential_claim_ids)
        for sorted_potential_claim_ids in df_tweet["top_claim_ids"]]

    # Evaluation - this requires 'is_myth' column to infer 'true_label'
    if args.do_eval:
        # Build true labels
        df_tweet["true_label"] = [1 if myth == "yes" else 0 for myth in df_tweet["is_myth"]]

        if args.balance_testing:
            df_tweet_yes = df_tweet[df_tweet["is_myth"]=="yes"]
            df_tweet_no = df_tweet[df_tweet["is_myth"]!="yes"]
            if df_tweet_yes.shape[0] < df_tweet_no.shape[0]:
                # Undersampling not-myth tweets
                df_tweet_no = df_tweet_no.sample(
                    n=df_tweet_yes.shape[0], random_state=args.random_seed)
            df_tweet_result = pd.concat([df_tweet_yes, df_tweet_no])
        else:
            df_tweet_result = df_tweet

        # Compute performance scores
        accuracy, precision, recall, f1 = accuracy_precision_recall_fscore_support(
            list(df_tweet_result["true_label"]), list(df_tweet_result["weak_label"]))
        print("Evaluation Performance")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)

        if args.save_dir:
            output_fp = f"{args.save_dir}/eval_results.txt"
            with open(output_fp, "w") as f:
                f.write("Evaluation Performance\n")
                f.write("Accuracy: {}\n".format(accuracy))
                f.write("Precision: {}\n".format(precision))
                f.write("Recall: {}\n".format(recall))
                f.write("F1: {}\n".format(f1))
            logger.info("Successfully saved results at: {}".format(output_fp))

    # Save the tweet dataframe with the prediction column 'weak_label'
    if args.save_dir:
        logger.info("DF Tweet Size: {}".format(df_tweet.shape))
        distribution_weak_label = df_tweet.groupby("weak_label").size()
        print(distribution_weak_label)

        # Save simple stats
        save_filepath_stats = f"{args.save_dir}/stats.txt"
        with open(save_filepath_stats, "w") as f:
            f.write(f"Total tweet size: {total_tweet_size}\n")
            if args.sample_size:
                f.write(f"Sampled tweet size: {sampled_tweet_size}\n")
            if "is_myth" in df_tweet.columns:
                f.write(f"Distribution: {distribution_is_myth}\n")
            f.write(f"Distribution: {distribution_weak_label}\n")

        if "true_label" in df_tweet.columns:
            save_columns = ["tweet_id", "text", "weak_label", "true_label"]
        else:
            save_columns = ["tweet_id", "text", "weak_label"]
        save_filepath_df_tweet = f"{args.save_dir}/predictions.csv"
        df_tweet[save_columns].to_csv(save_filepath_df_tweet, index=False)
        logger.info("Successfully saved predictions at: {}".format(save_filepath_df_tweet))


if __name__ == "__main__":
    main()