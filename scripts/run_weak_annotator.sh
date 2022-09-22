#!/bin/sh

UNLABELED_TWEET_FILEPATH="datasets/unlabeled_tweets/sample_unlabeled_tweets.csv"


echo ">>>>> Generating weak labels for: weather"
CLAIM_IDS_OF_INTEREST="135, 545, 586"
CUDA_VISIBLE_DEVICES=0 \
python weak_classifier.py \
    --input_filepath_tweet=$UNLABELED_TWEET_FILEPATH \
    --input_filepath_claim="datasets/claims/claims-covid-19.csv" \
    --similarity_model="sentence-transformers/bert-base-nli-stsb-mean-tokens" \
    --correct_claim_ids="$CLAIM_IDS_OF_INTEREST" \
    --top_k="10" \
    --save_dir="similarity_matrix/unlabeled_tweets/weather" \
    --sample_size=3000


echo ">>>>> Generating weak labels for: home_remedies"
CLAIM_IDS_OF_INTEREST="145, 243, 259, 283, 336, 360, 506, 583"
CUDA_VISIBLE_DEVICES=0 \
python weak_classifier.py \
    --input_filepath_tweet=$UNLABELED_TWEET_FILEPATH \
    --input_filepath_claim="datasets/claims/claims-covid-19.csv" \
    --similarity_model="sentence-transformers/bert-base-nli-stsb-mean-tokens" \
    --correct_claim_ids="$CLAIM_IDS_OF_INTEREST" \
    --top_k="10" \
    --save_dir="similarity_matrix/unlabeled_tweets/home_remedies" \
    --sample_size=3000