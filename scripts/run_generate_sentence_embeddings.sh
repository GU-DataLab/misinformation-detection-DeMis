#!/bin/sh


for myth_theme in "home_remedies" "weather"
do
    CUDA_VISIBLE_DEVICES=0 \
    python generate_sentence_embeddings.py \
        --input_tweet_filepath="datasets/tweets/weak_label_${myth_theme}_with_info.csv" \
        --output_filepath="embeddings/weak_label_$myth_theme.pt"
done