#!/usr/bin/env python
# coding: utf-8

'''
@title: Train DeMis Model
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: To train and save DeMis model.
'''


import os
import sys
import json
import time
import argparse
import traceback
import pandas as pd

import torch

from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from demis import DeMis
from libs.utils_data import MythTweetData
from libs.utils_misc import set_seed
from libs.pytorchtools import evaluate_model
from libs.utils_transformer import print_sequence_len_coverage_stats


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()
logger.info('There are {} GPUs.'.format(n_gpu))

if n_gpu > 1:
    raise NotImplementedError(f"Unsupport multiple GPUs. Found {n_gpu}. Please specify "
                               "to use only one GPU. E.g. use `CUDA_VISIBLE_DEVICES=0`.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_manual_filepath', type=str, required=True)
    parser.add_argument('--train_weak_filepath', type=str, required=True)
    parser.add_argument('--validation_filepath', type=str, required=True)
    parser.add_argument('--test_filepath', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--model_path', type=str, default="vinai/bertweet-covid19-base-cased")
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--sequence_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--bag_batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_bags', type=int, default=200)
    parser.add_argument('--log_level', type=str, default=None)
    parser.add_argument('--continue_from_checkpoint', type=str, default=None)
    parser.add_argument('--sentence_embedding_filepath', type=str, default=None)
    parser.add_argument('--strong_weak_ratio', type=float, default=None)
    parser.add_argument('--selector_lr', type=float, default=1e-4)
    args = parser.parse_args()

    if args.log_level:
        logger.remove()
        logger.add(sys.stderr, level=args.log_level)

    # Set random seed for reproducibility
    # set_seed(args.random_seed)

    # Whether use prob from weak labeler (sim score in our unsupervised method)
    # Should always be True
    args.use_weak_label_info = True
    if args.use_weak_label_info:
        state_size = 8
    else:
        state_size = 3

    # Load data from files
    train_manual_tweet_dataset = MythTweetData(
        pd.read_csv(args.train_manual_filepath, dtype={"tweet_id": str}))
    train_weak_tweet_dataset = MythTweetData(
        pd.read_csv(args.train_weak_filepath, dtype={"tweet_id": str}),
        use_weak_label_info=args.use_weak_label_info,
        use_tweet_ids=(args.sentence_embedding_filepath is not None))
    validate_tweet_dataset = MythTweetData(
        pd.read_csv(args.validation_filepath, dtype={"tweet_id": str}))
    test_tweet_dataset = MythTweetData(
        pd.read_csv(args.test_filepath, dtype={"tweet_id": str}))

    logger.info(f"Train Manual Data Size: {len(train_manual_tweet_dataset)}")
    logger.info(f"Train Weak Data Size: {len(train_weak_tweet_dataset)}")
    logger.info(f"Validation Data Size: {len(validate_tweet_dataset)}")
    logger.info(f"Test Data Size: {len(test_tweet_dataset)}")

    # Initialize DeMis model
    logger.info("Initializing DeMis")
    demis_model = DeMis(
        model_path=args.model_path,
        sequence_len=args.sequence_len,
        random_seed=args.random_seed,
        learning_rate=args.learning_rate,
        use_weak_label_info=args.use_weak_label_info,
        sentence_embedding_filepath=args.sentence_embedding_filepath,
        state_size=state_size)
    if args.continue_from_checkpoint:
        logger.info(f"Continue from checkpoint: {args.continue_from_checkpoint}")
        demis_model.load_models(args.continue_from_checkpoint, load_optimizers=True)

    # Check coverage of sequence length
    # E.g. if we set seq_len to 64, how many tweets are shorter than that.
    tmp_texts = list(train_manual_tweet_dataset.texts) + \
                list(train_weak_tweet_dataset.texts) + \
                list(validate_tweet_dataset.texts) + \
                list(test_tweet_dataset.texts)
    print_sequence_len_coverage_stats(
        tokenizer=demis_model.fake_news_detector.extractor.tokenizer,
        list_texts=tmp_texts,
        sequence_len=args.sequence_len,
        logger=logger)
    del tmp_texts

    # Create embedding indices dataset from text data
    logger.info("Converting texts to embedding indices and masks")
    # Manually labeled data
    train_manual_embedding_indices, train_manual_masks = demis_model.fake_news_detector.generate_emb_indices_and_mask(
        train_manual_tweet_dataset.texts)
    train_manual_dataset = TensorDataset(
        train_manual_embedding_indices,
        train_manual_masks,
        torch.tensor(train_manual_tweet_dataset.labels))
    # Weak-labeled data
    train_weak_embedding_indices, train_weak_masks = demis_model.fake_news_detector.generate_emb_indices_and_mask(
        train_weak_tweet_dataset.texts)
    if args.use_weak_label_info:
        if args.sentence_embedding_filepath:
            train_weak_dataset = TensorDataset(
                train_weak_embedding_indices,
                train_weak_masks,
                torch.tensor(train_weak_tweet_dataset.labels),
                torch.tensor(train_weak_tweet_dataset.weak_label_info),
                torch.LongTensor([int(x) for x in train_weak_tweet_dataset.tweet_ids]))
        else:
            train_weak_dataset = TensorDataset(
                train_weak_embedding_indices,
                train_weak_masks,
                torch.tensor(train_weak_tweet_dataset.labels),
                torch.tensor(train_weak_tweet_dataset.weak_label_info))
    else:
        train_weak_dataset = TensorDataset(
            train_weak_embedding_indices,
            train_weak_masks,
            torch.tensor(train_weak_tweet_dataset.labels))
    # Validation data
    validate_embedding_indices, validate_masks = demis_model.fake_news_detector.generate_emb_indices_and_mask(
        validate_tweet_dataset.texts)
    validate_dataset = TensorDataset(
        validate_embedding_indices,
        validate_masks,
        torch.tensor(validate_tweet_dataset.labels))
    # Test data
    test_embedding_indices, test_masks = demis_model.fake_news_detector.generate_emb_indices_and_mask(
        test_tweet_dataset.texts)
    test_dataset = TensorDataset(
        test_embedding_indices,
        test_masks,
        torch.tensor(test_tweet_dataset.labels))

    # Data Loader from PyTorch
    train_manual_loader = DataLoader(dataset=train_manual_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True)

    train_weak_loader = DataLoader(dataset=train_weak_dataset,
                                   batch_size=args.bag_batch_size,
                                   shuffle=True)

    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    # Manage output directory
    if args.output_dir.endswith("/"):
        args.output_dir = args.output_dir[:-1]

    start_training_time = time.time()
    checkpoint_prefix = str(int(start_training_time*1000) % 2**32)

    params = {
        "train_manual_loader": train_manual_loader,
        "train_weak_loader": train_weak_loader,
        "validate_loader": validate_loader,
        "test_loader": test_loader,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "num_bags": args.num_bags,
        "retrain_num_epochs": 1,
        "num_warmup_epochs": 0,
        "detection_train_epoch": 1,
        "checkpoint_prefix": checkpoint_prefix,
        "output_dir": args.output_dir,
        "no_selector": False,
        "static_baseline_performance": None,
        "use_only_target_selector_to_select": True,
        "load_pretrained_demis_models": None,
        "device": device,
        "minority_class": train_manual_tweet_dataset.get_minority_class(),
        "num_minority_class": train_manual_tweet_dataset.get_class_size(
            train_manual_tweet_dataset.get_minority_class()
        ),
        "strong_weak_ratio": args.strong_weak_ratio,
        "fix_baseline_performance": False,
        "selector_lr": args.selector_lr
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/{checkpoint_prefix}_args.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    logger.info("Start training DeMis")
    list_results = demis_model.train(**params)
    end_training_time = time.time()
    print("### Train Performance ###")
    for p in list_results:
        print(p)

    results = evaluate_model(
        demis_model.fake_news_detector, test_loader,
        demis_model.fake_news_detector.criterion, device,
        eval_pos_label_only=False, verbose=True)
    print("### Performance on test set ###")
    print(results)


    if args.output_dir:
        ###########################################
        # Evaluate best epoch validation F1 models
        ###########################################
        if demis_model.best_epoch_val_f1_models_dir:
            # Load the best epoch reward model
            best_demis_model = DeMis(
                demis_model.fake_news_detector.extractor,
                sequence_len=args.sequence_len,
                use_weak_label_info=args.use_weak_label_info,
                state_size=state_size)
            best_demis_model.load_models(
                demis_model.best_epoch_val_f1_models_dir, load_optimizers=True, device="cpu")

            # Get train results
            best_results_train = evaluate_model(
                best_demis_model.fake_news_detector, train_manual_loader,
                best_demis_model.fake_news_detector.criterion, device,
                eval_pos_label_only=False, verbose=True)
            print("### Best epoch validation f1 model performance on train set ###")
            print(best_results_train)
            with open(f"{demis_model.best_epoch_val_f1_models_dir}/best_epoch_val_f1_model_result_train.json", "w") as f:
                json.dump(best_results_train, f, indent=4)

            # Get validation results
            best_results_val = evaluate_model(
                best_demis_model.fake_news_detector, validate_loader,
                best_demis_model.fake_news_detector.criterion, device,
                eval_pos_label_only=False, verbose=True)
            print("### Best epoch validation f1 model performance on validation set ###")
            print(best_results_val)
            with open(f"{demis_model.best_epoch_val_f1_models_dir}/best_epoch_val_f1_model_result_val.json", "w") as f:
                json.dump(best_results_val, f, indent=4)

            # Get test results
            best_results_test = evaluate_model(
                best_demis_model.fake_news_detector, test_loader,
                best_demis_model.fake_news_detector.criterion, device,
                eval_pos_label_only=False, verbose=True)
            print("### Best epoch validation f1 model performance on test set ###")
            print(best_results_test)
            with open(f"{demis_model.best_epoch_val_f1_models_dir}/best_epoch_val_f1_model_result_test.json", "w") as f:
                json.dump(best_results_test, f, indent=4)


        #####################
        # Save meta data
        #####################
        with open(f"{args.output_dir}/{checkpoint_prefix}_running_time.txt", "w") as f:
            f.write(f"Total training time: {(end_training_time-start_training_time)/60:.3f} minutes")


if __name__=="__main__":
    try:
        start_time = time.time()

        # Run main function
        main()

        end_time = time.time()

        # Successfully done experiment
        success_text = "*" + __file__ + "*" + \
            " is finished after {} seconds with the following command.\n".format(
                round(end_time-start_time, 4))
        success_text += " ".join(sys.argv)
        logger.success(success_text)

    except Exception as e:
        end_time = time.time()

        print('Error after {} seconds of execution time'.format(
            round(end_time-start_time, 4)))
        print(traceback.format_exc())
