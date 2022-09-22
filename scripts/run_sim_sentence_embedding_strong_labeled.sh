#!/bin/sh

for myth_theme in "home_remedies" "weather"
do
    echo ">>>>> Processing $myth_theme"
    CUDA_VISIBLE_DEVICES=0 \
    python code/sim_sentence_embedding.py \
        --input_tweet_filepath="datasets/tweets/${myth_theme}_myth_train.csv" \
        --input_claim_filepath="datasets/claims/claims-covid-19.csv" \
        --output_dir="similarity_matrix/${myth_theme}/strong-labeled"
done