#!/bin/sh

python run_detector.py \
    --load_model_path="trained_models/misinfo_detector_covid_weather.pt" \
    --input_tweet_filepath="datasets/tweets/weather_myth_test.csv" \
    --output_filepath="predictions/weather_myth_test_with_prediction.csv" \
    --do_tweet_preprocessing \
    --device="cpu"