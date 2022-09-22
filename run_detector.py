#!/usr/bin/env python
# coding: utf-8

'''
@title: Run Detector
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: Run the fake news detector instance for fake news detection.
'''


import os
import csv
import tqdm
import torch
import argparse
import pandas as pd

from fake_news_detector import FakeNewsDetectorTransformer
from libs.utils_text import censor_text


def parse_arguments(parser):
    parser.add_argument('--base_model_path', type=str, default="vinai/bertweet-covid19-base-cased")
    parser.add_argument('--load_model_path', type=str, required=True)
    parser.add_argument('--input_tweet_filepath', type=str, required=True)
    parser.add_argument('--output_filepath', type=str, required=True)
    parser.add_argument('--do_tweet_preprocessing', action='store_true')
    parser.add_argument('--device', type=str, default="cpu")
    return parser


def main(args):
    # Init detector model
    fake_news_detector = FakeNewsDetectorTransformer(args.base_model_path)

    # Load model from checkpoint
    if torch.cuda.is_available():
        fake_news_detector.load_state_dict(torch.load(args.load_model_path))
    else:
        fake_news_detector.load_state_dict(
            torch.load(args.load_model_path, map_location=torch.device('cpu')))

    # Check output directory
    output_dir = "/".join(args.output_filepath.split("/")[:-1])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load to device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fake_news_detector = fake_news_detector.to(device)

    # Load input tweets
    df = pd.read_csv(args.input_tweet_filepath)
    print(f"Tweet size: {df.shape[0]}")

    if args.do_tweet_preprocessing:
        df["text"] = [censor_text(x) for x in tqdm.tqdm(df["text"])]

    # Predict misinformation tweets
    df["prediction"] = fake_news_detector.predict(list(df["text"]), device).tolist()

    # Save output
    print(f"Saving output to: {args.output_filepath}... ", end="")
    df.to_csv(args.output_filepath,\
        escapechar='\"', \
        quotechar='\"',\
        quoting=csv.QUOTE_ALL,\
        index=False)
    print("Done")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    args = parser.parse_args()
    main(args)