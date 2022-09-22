#!/bin/sh

echo ">>>>> Balancing weak labels for: weather"
python get_balanced_weak_labeled_data.py \
    --input_dir="similarity_matrix/unlabeled_tweets/weather" \
    --output_filepath="datasets/tweets/weak_label_weather.csv" \
    --each_class_size=60


echo ">>>>> Balancing weak labels for: home_remedies"
python get_balanced_weak_labeled_data.py \
    --input_dir="similarity_matrix/unlabeled_tweets/home_remedies" \
    --output_filepath="datasets/tweets/weak_label_home_remedies.csv" \
    --each_class_size=200