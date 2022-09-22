#!/usr/bin/env python
# coding: utf-8

'''
@title: Textual Feature Extractor
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: Textual Feature Extractor from DeMis model.
'''


import sys
import tqdm
import random
import argparse
import numpy as np

import torch
import torch.nn as nn

from loguru import logger
from transformers import AutoTokenizer, AutoModel, AutoConfig


class TextualFeatureExtractor(nn.Module):
    def __init__(self, model_path: str, freeze_model=False):
        super(TextualFeatureExtractor, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, normalization=True)
        self.embedding_model = AutoModel.from_pretrained(model_path, add_pooling_layer=False)
        self.config = AutoConfig.from_pretrained(model_path)

        if freeze_model:
            for param in self.embedding_model.parameters():
                param.requires_grad = False


    def forward(self, input_tuple):
        input_ids, attention_mask = input_tuple
        features = self.embedding_model(
            input_ids=input_ids, attention_mask=attention_mask)
        # return mean_pooling(features, mask)
        return features


    def generate_emb_indices_and_mask(self, texts, padding="max_length",
                                      max_seq_length=128):
        if self.tokenizer.model_max_length < max_seq_length:
            logger.warning(
                f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
                f"model ({self.tokenizer.model_max_length}). "
                f"Using max_seq_length={self.tokenizer.model_max_length} instead."
            )
        max_seq_length = min(max_seq_length, self.tokenizer.model_max_length)
        result = self.tokenizer(
            texts, padding=padding, max_length=max_seq_length, truncation=True)
        embedding_indices = torch.tensor(result["input_ids"])
        mask = torch.tensor(result["attention_mask"])
        return embedding_indices, mask


def set_seed(k):
    random.seed(k)
    np.random.seed(k)
    torch.manual_seed(k)


def parse_arguments(parser):
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_filepath', type=str, default=None)
    parser.add_argument('--max_seq_length', type=int, default=28)
    parser.add_argument('--batch_size', type=int, default=128)
    return parser


def main(args):
    set_seed(k=3407)

    print('Loading data')
    if args.input_filepath:
        raise NotImplementedError
    else:
        texts = ["I love you so much."] * 3

    print('Loading extractor model')
    extractor = TextualFeatureExtractor(model_path=args.model_path)

    # Forward pass through CNN
    embedding_indices, mask = extractor.generate_emb_indices_and_mask(
        texts, padding="max_length", max_seq_length=128)

    print("Embedding Indices")
    print(embedding_indices)
    print("Mask")
    print(mask)

    if torch.cuda.is_available():
        extractor.cuda()
        embedding_indices = embedding_indices.to("cuda")
        mask = mask.to("cuda")
    input_tuple = (embedding_indices, mask)
    embedding_vectors = extractor(input_tuple)
    print("Embeddings")
    # print(embedding_vectors.shape)
    print(embedding_vectors)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    args = parser.parse_args()
    main(args)