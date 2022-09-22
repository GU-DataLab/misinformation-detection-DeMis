#!/bin/sh


INPUT_DIR="datasets/tweets"

for myth_theme in "home_remedies" "weather"
do
    # Strong-labeled tweets
    python code/build_info_for_RL_state.py \
        --input_filepath="$INPUT_DIR/${myth_theme}_myth_train.csv" \
        --output_filepath="$INPUT_DIR/${myth_theme}_myth_train_with_info.csv" \
        --similarity_matrix_filepath="similarity_matrix/$myth_theme/strong-labeled/sim-bert-base-nli-stsb-mean-tokens-mean-cleaned.json"

    # Weak-labeled tweets
    python code/build_info_for_RL_state.py \
        --input_filepath="$INPUT_DIR/weak_label_${myth_theme}.csv" \
        --output_filepath="$INPUT_DIR/weak_label_${myth_theme}_with_info.csv" \
        --similarity_matrix_filepath="similarity_matrix/unlabeled_tweets/$myth_theme/sim-sentence-transformers-bert-base-nli-stsb-mean-tokens-mean_cleaned.json"
done