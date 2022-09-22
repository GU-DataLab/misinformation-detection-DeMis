#!/usr/bin/env python
# coding: utf-8

'''
@title: Similarity Sentence Embedding
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: Applying sentence-transformers similarity algorithms to compare tweets and claims
'''


from tqdm import tqdm, trange
from loguru import logger
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModel

import os
import sys
import torch
import argparse

import numpy as np
import pandas as pd

from libs.utils_text import clean_text
from libs.utils_similarity import save_similarity_matrix


# Specify GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info('There are {} GPUs.'.format(n_gpu))

if n_gpu > 1:
    raise NotImplementedError(f"Unsupport multiple GPUs. Found {n_gpu}. Please specify "
                               "to use only one GPU. E.g. use `CUDA_VISIBLE_DEVICES=0`.")


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# Max Pooling - Take the max value over time for every dimension
def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    max_over_time = torch.max(token_embeddings, 1)[0]
    return max_over_time


class SentenceEncoder():
    def __init__(self, model_path='bert-base-uncased', use_gpu=True):
        # Load AutoModel from huggingface model repository
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.n_gpu = n_gpu if use_gpu else 0
        if self.n_gpu > 0:
            self.model.cuda()  # Load model to GPU

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

    def get_embeddings(self, sentences, pooling="cls", batch_size=64, verbose=False):
        if self.n_gpu == 0:
            batch_size = 1
        
        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

        # Prepare batch
        # print(encoded_input)
        tuple_encoded_input = ()
        for k in ["input_ids" ,"token_type_ids", "attention_mask"]:
            if k in encoded_input.keys():
                tuple_encoded_input = tuple_encoded_input + (encoded_input[k],)

        data = TensorDataset(*tuple_encoded_input)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

        # Generate embeddings
        all_sentence_embeddings = ()
        if verbose:
            dataloader = tqdm(dataloader)
        for batch in dataloader:
            # Add batch to GPU
            if self.n_gpu > 0:
                batch = tuple(t.to(device) for t in batch)

            # Compute embeddings for this specific batch
            with torch.no_grad():
                # Unpack the inputs from our dataloader
                if {"input_ids", "token_type_ids", "attention_mask"} == set(encoded_input.keys()):
                    b_input_ids, b_token_type_ids, b_attention_mask = batch
                    b_input_support = {"token_type_ids": b_token_type_ids, "attention_mask": b_attention_mask}
                elif {"input_ids", "token_type_ids"} == set(encoded_input.keys()):
                    b_input_ids, b_token_type_ids = batch
                    b_input_support = {"token_type_ids": b_token_type_ids}
                elif {"input_ids", "attention_mask"} == set(encoded_input.keys()):
                    b_input_ids, b_attention_mask = batch
                    b_input_support = {"attention_mask": b_attention_mask}
                elif {"input_ids"} == set(encoded_input.keys()):
                    b_input_ids = batch
                    b_input_support = {}

                # Forward pass
                model_output = self.model(b_input_ids, **b_input_support)

            # Perform pooling
            if pooling == "mean":
                sentence_embeddings = mean_pooling(model_output, b_attention_mask)
            elif pooling == "max":
                sentence_embeddings = max_pooling(model_output, b_attention_mask)
            elif pooling == "cls":
                sentence_embeddings = model_output[0][:,0] # Take the first token ([CLS]) from each sentence
            else:
                raise ValueError("Pooling is invalid.")

            # Append to the output tuple
            all_sentence_embeddings = all_sentence_embeddings + (sentence_embeddings,)

        # Return a list of tensors
        return torch.cat(all_sentence_embeddings, dim=0)


def get_similarity_matrix(sentence_embeddings_A, sentence_embeddings_B):
    cosine = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
    sim_scores = np.array([[0 for j in range(sentence_embeddings_B.shape[0])] for i in range(sentence_embeddings_A.shape[0])]).astype(float)
    for i in trange(sentence_embeddings_A.shape[0], desc="Compute Score"):
        for j in range(sentence_embeddings_B.shape[0]):
            input1 = sentence_embeddings_A[i]
            input2 = sentence_embeddings_B[j]
            sim_scores[i][j] = cosine(input1, input2).item()
    return sim_scores


def parse_arguments(parser):
    parser.add_argument('--input_tweet_filepath', type=str, required=True)
    parser.add_argument('--input_claim_filepath', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser


def main(args):
    # Set logger level to print out
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Tweet input
    df_tweet = pd.read_csv(args.input_tweet_filepath, dtype={"tweet_id": str})
    logger.info(f"Total tweet: {df_tweet.shape[0]}")

    # Claim input
    df_claim = pd.read_csv(args.input_claim_filepath)
    logger.info(f"Total claim: {df_claim.shape[0]}")

    # Clean tweets and claims
    df_tweet['text_cleaned'] = df_tweet["text"].apply(lambda x: clean_text(x))
    df_claim["claim_cleaned"] = df_claim["claim"].apply(lambda x: clean_text(x))

    # NOTE: Specify models to use
    model_to_pooling = {
        "bert-base-nli-stsb-mean-tokens": ["mean"],
    }

    # # All models
    # model_to_pooling = {
    #     "nli-bert-base": ["mean"],
    #     "nli-bert-base-max-pooling": ["max"],
    #     "nli-bert-base-cls-pooling": ["cls"],
    #     "bert-base-nli-stsb-mean-tokens": ["mean"],
    #     "bert-base-nli-mean-tokens": ["mean"],
    #     "bert-base-nli-max-tokens": ["max"],
    #     "bert-base-nli-cls-token": ["cls"],
    #     "bert-large-nli-stsb-mean-tokens": ["mean"],
    #     "bert-large-nli-mean-tokens": ["mean"],
    #     "bert-large-nli-max-tokens": ["max"],
    #     "bert-large-nli-cls-token": ["cls"],
    #     "stsb-bert-base": ["mean"],
    #     "stsb-bert-large": ["mean"],
    #     "stsb-roberta-base-v2": ["mean"],
    #     "stsb-roberta-large": ["mean"],
    #     "nli-roberta-base-v2": ["mean"],
    #     "nli-roberta-large": ["mean"],
    #     "roberta-base-nli-stsb-mean-tokens": ["mean"],
    #     "roberta-base-nli-mean-tokens": ["mean"],
    #     "roberta-large-nli-stsb-mean-tokens": ["mean"],
    #     "roberta-large-nli-mean-tokens": ["mean"],
    # }

    # Validate model_to_pooling
    possible_poolings = {"mean", "max", "cls"}
    for model_name, vals in model_to_pooling.items():
        # Validate pooling methods
        for v in vals:
            if v not in possible_poolings:
                raise ValueError(f"Invalid pooling '{v}' at model '{model_name}' with the poolings {vals}.")


    # Generate sentence embeddings
    cleans = ["_cleaned"]  # NOTE: Select whether use clean data or not
    # cleans = ["", "_cleaned"]  # Select whether use clean data or not
    for model_name in model_to_pooling:
        # output_dir = f"{args.output_dir}/{model_name}"
        output_dir = args.output_dir
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Run encoder
        poolings = model_to_pooling[model_name]
        for pooling in poolings:
            model_path = "sentence-transformers/" + model_name
            logger.info(f"Load Model: {model_path}")
            encoder = SentenceEncoder(model_path)

            for clean in cleans:
                sentence_embeddings_A = encoder.get_embeddings(df_tweet[f"text{clean}"].tolist(), pooling=pooling)
                sentence_embeddings_B = encoder.get_embeddings(df_claim[f"claim{clean}"].tolist(), pooling=pooling)

                logger.info(f"Sentence Embeddings A: {sentence_embeddings_A.shape}")
                logger.info(f"Sentence Embeddings B: {sentence_embeddings_B.shape}")

                sim_scores = get_similarity_matrix(sentence_embeddings_A, sentence_embeddings_B)

                logger.info(f"Similarity Matrix Size: {sim_scores.shape}")

                # Save to JSON file
                clean = clean.replace("_", "-")
                output_fp = f"{output_dir}/sim-{model_name}-{pooling}{clean}.json"
                save_similarity_matrix(sim_scores, filename=output_fp, query_ids=list(df_tweet["tweet_id"]))
                logger.info(f"Saved similiarities at: {output_fp}")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    args = parser.parse_args()
    main(args)