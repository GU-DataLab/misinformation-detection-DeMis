#!/bin/sh


CUDA_VISIBLE_DEVICES=0 \
python -u code/train_demis.py \
    --train_manual_filepath="datasets/tweets/weather_myth_train_with_info.csv" \
    --train_weak_filepath="datasets/tweets/weak_label_weather_with_info.csv" \
    --validation_filepath="datasets/tweets/weather_myth_val.csv" \
    --test_filepath="datasets/tweets/weather_myth_test.csv" \
    --output_dir="trained_models/weather" \
    --sentence_embedding_filepath="embeddings/weak_label_weather.pt" \
    --learning_rate=1e-5 \
    --bag_batch_size=8 \
    --num_epochs=2 \
    --num_bags=5