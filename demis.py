#!/usr/bin/env python
# coding: utf-8

'''
@title: DeMis
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: The DeMis model
'''


import os
import copy
import json
import time
from typing import Tuple
import tqdm
import torch
import random
import shutil

from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from fake_news_detector import FakeNewsDetectorTransformer
from reinforced_selector import ReinforcedSelector, STATE_SIZE
# from libs.utils_misc import set_seed
from libs.pytorchtools import train_model, evaluate_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()
logger.info('There are {} GPUs.'.format(n_gpu))

if n_gpu > 1:
    raise NotImplementedError(f"Unsupport multiple GPUs. Found {n_gpu}. Please specify "
                               "to use only one GPU. E.g. use `CUDA_VISIBLE_DEVICES=0`.")

POSSIBLE_LABELS = [0, 1]
POSSIBLE_REWARD_METRICS = ["accuracy", "precision", "recall", "f1"]


class DeMis(object):
    def __init__(self, model_path,
                 sequence_len=128,
                 use_weak_label_info=False,
                 sentence_embedding_filepath=None,
                 state_size=3,
                 learning_rate=0.00001,
                 random_seed=None):
        """ DeMis model.

        Args:
            sentence_embedding_filepath (str): A path to the file containing embeddings from
                sentence transformer. Will be used to find max similarity in a batch during
                state creation.
        """

        super(DeMis, self).__init__()

        # Params
        self.random_seed = random_seed
        self.sequence_len = sequence_len
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.model_path = model_path
        self.use_weak_label_info = use_weak_label_info
        self.use_sentence_transformer = sentence_embedding_filepath is not None

        if self.use_weak_label_info and self.state_size != STATE_SIZE["with_info"]:
            raise ValueError(
                f"If use_weak_label_info, state size must be"
                f" {STATE_SIZE['with_info']}, found {self.state_size}.")
        if not self.use_weak_label_info and self.state_size != STATE_SIZE["without_info"]:
            raise ValueError(
                f"If NOT use_weak_label_info, state size must be"
                f" {STATE_SIZE['without_info']}, found {self.state_size}.")

        # Initialize neural networks
        self.fake_news_detector = FakeNewsDetectorTransformer(
            model_path=self.model_path)
        self.reinforced_selector = ReinforcedSelector(
            self.fake_news_detector, hidden_size=8, state_size=state_size)

        if sentence_embedding_filepath:
            ReinforcedSelector.load_global_sentence_embeddings(
                sentence_embedding_filepath,
                device=device)

        # Initialize optimizers and loss function
        self.optimizer_fake_news_detector = torch.optim.AdamW(
            self.fake_news_detector.parameters(), lr=self.learning_rate)
        self.optimizer_selector = torch.optim.AdamW(
            self.reinforced_selector.parameters(), lr=self.learning_rate)

        # Store best model paths
        self.best_epoch_val_f1_models_dir = None


    def __str__(self):
        return "\n".join([
            str(self.reinforced_selector),
            str(self.fake_news_detector)])


    def train(self, train_manual_loader: DataLoader, train_weak_loader: DataLoader,
              validate_loader: DataLoader, batch_size: int, num_epochs: int, checkpoint_prefix,
              num_bags: int, test_loader: DataLoader = None, detection_train_epoch: int = 1,
              retrain_num_epochs=1, num_warmup_epochs=0, output_dir: str = None, early_stop=-3,
              reward_metric="f1", no_selector=False, static_baseline_performance=None,
              use_only_target_selector_to_select=False, num_pretrain_detector_epochs=20,
              load_pretrained_demis_models=None, device=None, minority_class=None,
              num_minority_class=None, strong_weak_ratio=None, fix_baseline_performance=False,
              selector_lr=1e-4):
        """ Train the DeMis model

        This function trains the DeMis model and save the best models and
        performance results.

        Args:
            test_loader (DataLoader): To keep monitoring the performance every epoch.
                Default is None indicating no run testing.
            num_warmup_epochs (int): Number of epoch to train the detector with
                the manually-labeled samples only. The RL selector is still trained
                but its selected weak-labeled samples will not be used for training
                the fake news detector.
            early_stop (int): Define number of epoch for early stopping when train
                the fake news detector model. Negative value means that validation
                loss is used to determine the stopping epoch. Positive value means
                accuracy is used instead.
            reward_metric (str): Metric to compute reward. Can be accuracy, precision,
                recall and f1.
            no_selector (bool): Whether to use RL selector to select weak-labeled samples
                or randomly select.
            use_only_target_selector_to_select (bool): Whether to use only target selector
                to select retained samples. They are used to re-train the target
                detector model.
            strong_weak_ratio (float): Strong-weak label training ratio, E.g. 2.0 means two times of minority
                class size will be drawn from selected samples.
            fix_baseline_performance (bool): Always use the baseline performance from the trained model
                using the manually-labeled training data.
        """

        self.minority_class = minority_class
        self.num_minority_class = num_minority_class
        logger.info(f"Minority Class: {self.minority_class}")
        logger.info(f"Minority Class Size: {self.num_minority_class}")
        logger.info(f"fix_baseline_performance: {fix_baseline_performance}")
        logger.info(f"selector_lr: {selector_lr}")
        list_results = []

        if no_selector and num_bags > 0:
            raise ValueError(f"If no_selector=True, num_bags must be positive " \
            "integer. Found {num_bags}.")

        if reward_metric and reward_metric not in POSSIBLE_REWARD_METRICS:
            raise ValueError(f"Given `reward_metric` is invalid. Found {reward_metric}.")

        start_time = time.time()
        # checkpoint_prefix = str(int(start_time*1000) % 2**32)

        # Use GPU if not provided and if GPU is available
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Use current directory to save models and results if not provided
        if output_dir is None:
            output_dir = str(os.path.abspath(os.getcwd()))
        output_dir = output_dir.strip("/")

        if load_pretrained_demis_models:
            logger.info("Load pre-trained models")
            self.load_models(load_pretrained_demis_models,
                             load_optimizers=False, device=device)
            baseline_results = evaluate_model(
                self.fake_news_detector, validate_loader,
                self.fake_news_detector.criterion, device=device,
                eval_pos_label_only=False, verbose=False,
                minority_class=self.minority_class)
            baseline_performance = baseline_results[minority_class][reward_metric]
            logger.info(f"Baseline {reward_metric} from pre-train detector: {baseline_performance}")
            if baseline_performance <= 0:
                raise ValueError(f"Baseline performance should be more than zero.")
        else:
            # Pre-train fake news detector
            if static_baseline_performance is None:
                baseline_results = train_model(
                    self.fake_news_detector, train_manual_loader, validate_loader,
                    num_epochs=num_pretrain_detector_epochs, optimizer=self.optimizer_fake_news_detector,
                    criterion=self.fake_news_detector.criterion, load_best_model=True,
                    early_stop=None, device=device, eval_pos_label_only=False,
                    all_possible_labels=POSSIBLE_LABELS, verbose=False, eval_warning="ignore",
                    best_model_measure="f1", minority_class=self.minority_class
                )
                baseline_performance = baseline_results[minority_class][f"val_{reward_metric}"]
                logger.info(f"Baseline {reward_metric} from pre-train detector: {baseline_performance}")
                if baseline_performance <= 0:
                    raise ValueError(f"Baseline performance should be more than zero.")

            # Pre-train the selector by fixing the detector model
            logger.info("Pre-train reinforced selector")
            self.reinforced_selector, self.fake_news_detector = self._train_selector(
                train_manual_loader=train_manual_loader, train_weak_loader=train_weak_loader,
                validate_loader=validate_loader, test_loader=test_loader,
                num_epochs=num_epochs, num_bags=num_bags,
                static_baseline_performance=static_baseline_performance,
                num_warmup_epochs=num_warmup_epochs, checkpoint_prefix=checkpoint_prefix,
                output_dir=output_dir, retrain_num_epochs=retrain_num_epochs,
                detection_train_epoch=detection_train_epoch, fix_detector=True,
                no_selector=no_selector, batch_size=batch_size, reward_metric=reward_metric,
                early_stop=early_stop, use_only_target_selector_to_select=use_only_target_selector_to_select,
                device=device,
                selector_lr=selector_lr,
                strong_weak_ratio=strong_weak_ratio)
            # Save pre-trained models
            tmp_output_dir = f"{output_dir}/{checkpoint_prefix}_pretrained_model"
            self.save_models(output_dir=tmp_output_dir, save_optimizers=False)
            self.reinforced_selector.fake_news_detector = self.fake_news_detector

        if static_baseline_performance:
            baseline_performance = static_baseline_performance

        # Reset fake news detector
        self.fake_news_detector = FakeNewsDetectorTransformer(
            model_path=self.model_path,)
        self.optimizer_fake_news_detector = torch.optim.AdamW(
            self.fake_news_detector.parameters(), lr=self.learning_rate)
        self.reinforced_selector.fake_news_detector = self.fake_news_detector
        # Warmup a bit
        train_model(
            self.fake_news_detector, train_manual_loader, validate_loader,
            num_epochs=int(num_pretrain_detector_epochs), optimizer=self.optimizer_fake_news_detector,
            stop_when_metric_change="f1", baseline_matric_value=0.0,
            criterion=self.fake_news_detector.criterion, load_best_model=True,
            early_stop=None, device=device, eval_pos_label_only=False,
            all_possible_labels=POSSIBLE_LABELS, verbose=False, eval_warning="ignore",
            best_model_measure="f1", minority_class=self.minority_class
        )

        # Jointly train them
        logger.info("Jointly train reinforced selector and detector")
        self.reinforced_selector, self.fake_news_detector = self._train_selector(
            train_manual_loader=train_manual_loader, train_weak_loader=train_weak_loader,
            validate_loader=validate_loader, test_loader=test_loader,
            num_epochs=num_epochs, num_bags=num_bags,
            static_baseline_performance=static_baseline_performance,
            num_warmup_epochs=num_warmup_epochs, checkpoint_prefix=checkpoint_prefix,
            output_dir=output_dir, retrain_num_epochs=retrain_num_epochs,
            detection_train_epoch=detection_train_epoch, fix_detector=False,
            no_selector=no_selector, batch_size=batch_size, reward_metric=reward_metric,
            early_stop=early_stop, use_only_target_selector_to_select=use_only_target_selector_to_select,
            best_epoch_val_f1=0,
            device=device,
            selector_lr=selector_lr,
            strong_weak_ratio=strong_weak_ratio,
            fix_baseline_performance=baseline_performance if fix_baseline_performance else False)
        self.reinforced_selector.fake_news_detector = self.fake_news_detector

        # Remove temp models
        shutil.rmtree(f"{output_dir}/{checkpoint_prefix}_pretrained_model")

        end_time = time.time()
        logger.info(f"Training DeMis time: {(end_time-start_time)/60:.3f} minutes")

        return list_results


    def _train_selector(
        self, train_manual_loader, train_weak_loader, validate_loader, test_loader,
        num_epochs, num_bags, static_baseline_performance,
        num_warmup_epochs, checkpoint_prefix, output_dir, retrain_num_epochs,
        detection_train_epoch, fix_detector, no_selector, batch_size, reward_metric,
        early_stop, use_only_target_selector_to_select,
        best_epoch_val_f1=0,
        device=None, strong_weak_ratio=None,
        fix_baseline_performance=False, selector_lr=1e-4
        ) -> Tuple[ReinforcedSelector, FakeNewsDetectorTransformer]:

        start_time = time.time()

        # Tmp reinforced selector to jointly select samples with the target selector
        # at the end of each epoch.
        tmp_reinforced_selector = copy.deepcopy(self.reinforced_selector)
        tmp_optimizer_selector = torch.optim.AdamW(
            tmp_reinforced_selector.parameters(), lr=2*selector_lr if fix_detector else selector_lr)
            # tmp_reinforced_selector.parameters(), lr=2e-5 if fix_detector else 1e-5)

        # Put models to device
        self.reinforced_selector = self.reinforced_selector.to(device)
        tmp_reinforced_selector = tmp_reinforced_selector.to(device)

        # Train
        for epoch in range(num_epochs):
            bag_counter = 0
            total_epoch_reward = 0
            done_epoch = False

            # Backup original states of models before RL training loop
            target_fake_news_detector_state_dict = copy.deepcopy(
                self.fake_news_detector.state_dict())
            target_optimizer_fake_news_detector_state_dict = copy.deepcopy(
                self.optimizer_fake_news_detector.state_dict())

            # Reset gradients
            self.optimizer_fake_news_detector.zero_grad()
            self.optimizer_selector.zero_grad()
            tmp_optimizer_selector.zero_grad()

            # Get baseline metric dynamically
            if static_baseline_performance is None and reward_metric != "loss":
                static_baseline_performance = float("-inf")
            baseline_results = evaluate_model(
                self.fake_news_detector, validate_loader,
                self.fake_news_detector.criterion, device=device,
                eval_pos_label_only=False, verbose=False,
                minority_class=self.minority_class)
            if reward_metric == "loss":
                raise ValueError("Reward metric `loss` is invalid.")
            if fix_baseline_performance:
                baseline_performance = fix_baseline_performance
            else:
                if reward_metric == "accuracy":
                    baseline_performance = max(
                        static_baseline_performance,
                        baseline_results[reward_metric])
                else:
                    baseline_performance = max(
                        static_baseline_performance,
                        baseline_results[self.minority_class][reward_metric])  # consider class label 1

            # Train bags
            bag_pbar = tqdm.tqdm(total=num_bags, desc=f"Epoch {epoch}/{num_epochs}")
            while bag_counter < num_bags and not done_epoch:
                for weak_data in train_weak_loader:

                    if self.use_weak_label_info:
                        if self.use_sentence_transformer:
                            train_inputs_weak, train_masks_weak, train_labels_weak, train_infos_weak, train_tweet_ids_weak = weak_data
                        else:
                            train_inputs_weak, train_masks_weak, train_labels_weak, train_infos_weak = weak_data
                            train_tweet_ids_weak = None
                    else:
                        if self.use_sentence_transformer:
                            raise ValueError("If NOT use weak label info, "
                                             "use_sentence_transformer is not avaliable.")
                        train_inputs_weak, train_masks_weak, train_labels_weak = weak_data
                        train_infos_weak = None

                    if device.type == 'cuda':
                        train_inputs_weak = train_inputs_weak.to(device)
                        train_masks_weak = train_masks_weak.to(device)
                        train_labels_weak = train_labels_weak.to(device)

                    # Forward the reinforced selector
                    # Sample actions
                    with torch.no_grad():
                        self.reinforced_selector(
                            train_inputs_weak, train_masks_weak, train_labels_weak,
                            infos_from_weak_labeler=train_infos_weak,
                            decide_actions=(not fix_detector),
                            tweet_ids=train_tweet_ids_weak, device=device)

                    # Get selected indices of samples AFTER forward pass the
                    # reinforced selector. It will update environment.
                    selected_sample_indices = self.reinforced_selector.get_selected_sample_indices()
                    selected_size = len(selected_sample_indices)

                    if selected_size == 0:
                        # Update variables
                        bag_counter += 1
                        bag_pbar.update(1)
                        if bag_counter >= num_bags:
                            done_epoch = True
                            break
                        continue

                    # Select weak-labeled samples by selected indices
                    selected_train_inputs_weak = train_inputs_weak[selected_sample_indices].detach().clone() if selected_size > 0 else torch.tensor([])
                    selected_train_masks_weak = train_masks_weak[selected_sample_indices].detach().clone() if selected_size > 0 else torch.tensor([])
                    selected_train_labels_weak = train_labels_weak[selected_sample_indices].detach().clone() if selected_size > 0 else torch.tensor([])

                    # Build dataset for re-training
                    retrain_dataset = TensorDataset(selected_train_inputs_weak,
                                                    selected_train_masks_weak,
                                                    selected_train_labels_weak)
                    # set_seed(epoch * self.random_seed)  # to generate different results every epoch
                    retrain_dataloader = DataLoader(
                        dataset=retrain_dataset, batch_size=batch_size, shuffle=True)

                    # Re-train fake news detector
                    results = train_model(
                        self.fake_news_detector, retrain_dataloader,
                        validate_loader, retrain_num_epochs,
                        self.optimizer_fake_news_detector,
                        self.fake_news_detector.criterion,
                        load_best_model=False, early_stop=None,
                        eval_pos_label_only=False, device=device, verbose=False,
                        eval_warning="ignore", minority_class=self.minority_class)

                    # Lastest val accuracy (consider class label 1)
                    if reward_metric == "accuracy":
                        current_bag_performance = results[f"val_{reward_metric}"]
                    else:
                        current_bag_performance = results[self.minority_class][f"val_{reward_metric}"]

                    # Compute reward
                    if current_bag_performance == baseline_performance:
                        # 10% of baseline performance if current perf is similar
                        # This helps tweak the select if the model always predict one
                        # class at the booststraping step.
                        reward = -(0.1 * baseline_performance)
                    else:
                        reward = current_bag_performance - baseline_performance
                    # Make it percentage
                    if reward_metric in ["precision", "recall", "f1"]:
                        reward = reward * 100
                    # If reward is too small, amplify it
                    if abs(reward) < 1e-6:  # original is 1e-9
                        reward = total_epoch_reward / (bag_counter + 1)
                    # Store total reward
                    total_epoch_reward += reward

                    # Compute policy values from tmp selector
                    policy_values = tmp_reinforced_selector(
                        train_inputs_weak, train_masks_weak, train_labels_weak,
                        infos_from_weak_labeler=train_infos_weak, decide_actions=True,
                        tweet_ids=train_tweet_ids_weak, device=device)

                    # Compute policy objective function
                    policy_reward = torch.sum(torch.log(policy_values) * reward)  # maximize this
                    policy_loss = -1 * policy_reward  # but minimize this

                    # Update reinforced selector model
                    tmp_reinforced_selector = tmp_reinforced_selector.to(device)
                    policy_loss.backward()
                    tmp_optimizer_selector.step()
                    tmp_optimizer_selector.zero_grad()

                    # Save results every bag but not save model
                    dict_performance = {
                        "epoch": epoch,
                        "bag": bag_counter,
                        "current_bag_performance": current_bag_performance,
                        "baseline_performance": baseline_performance,
                        "reward": reward,
                        "policy_reward": policy_reward.detach().item(),
                        "total_epoch_reward": total_epoch_reward,
                        "results": results,
                        "selected_size": selected_size,
                        "selected_sample_indices": selected_sample_indices
                    }

                    # Use the detector that is NOT retrained by selected samples
                    self.fake_news_detector.load_state_dict(
                        target_fake_news_detector_state_dict)
                    self.optimizer_fake_news_detector.load_state_dict(
                        target_optimizer_fake_news_detector_state_dict)

                    # Update the detector of selector to the old one
                    tmp_reinforced_selector.fake_news_detector = self.fake_news_detector
                    self.reinforced_selector.fake_news_detector = self.fake_news_detector

                    # Update variables
                    bag_counter += 1
                    bag_pbar.update(1)
                    if bag_counter >= num_bags:
                        done_epoch = True
                        break

            # End processing bag level
            bag_pbar.close()

            # Free memory
            if num_bags > 0:
                del train_inputs_weak
                del train_masks_weak
                del train_labels_weak
            torch.cuda.empty_cache()

            # Update target selector slowly
            self.update_target_policy_network(
                current_reinforced_selector_params=copy.deepcopy(
                    tmp_reinforced_selector.state_dict()),
                target_detector_params=None,
                learning_rate=0.001,
                device=device)

            # Bring back epoch-level states of models
            self.fake_news_detector.load_state_dict(
                target_fake_news_detector_state_dict)
            self.optimizer_fake_news_detector.load_state_dict(
                target_optimizer_fake_news_detector_state_dict)
            # MUST load this after load back fake news detector
            self.reinforced_selector.fake_news_detector = self.fake_news_detector

            # Select samples using the new selector and then combine selected
            # samples and manually-labeled data to train the detection model.
            combined_inputs = []
            combined_masks = []
            combined_labels = []
            combined_tweet_ids = []

            # Combine selected samples
            total_batch_selected_sample_indices = []
            total_batch_selected_size = 0
            is_done_selection = False
            if num_warmup_epochs > 0:  # still warmup with no selected samples
                num_warmup_epochs -= 1
            elif not fix_detector:  # use selected samples in the detector training
                for idx, batch in tqdm.tqdm(enumerate(train_weak_loader), total=len(train_weak_loader), desc="Selecting samples"):
                    # Avoid over-train on weak-labeled by balancing the same size
                    # as manually-labeled data
                    if strong_weak_ratio and not is_done_selection and total_batch_selected_size >= int(self.num_minority_class*strong_weak_ratio):
                        # Sampling selected data if too many selected
                        num_samples = int(self.num_minority_class*strong_weak_ratio)
                        tmp_p = torch.tensor([1.0/total_batch_selected_size for _ in range(total_batch_selected_size)])
                        sampled_indices = tmp_p.multinomial(num_samples=num_samples, replacement=False)

                        tmp_total_batch_selected_sample_indices = [t for l in total_batch_selected_sample_indices for t in l]
                        total_batch_selected_sample_indices = [torch.tensor(tmp_total_batch_selected_sample_indices, device="cpu")[sampled_indices].tolist()]
                        combined_inputs = [torch.cat(combined_inputs)[sampled_indices]]
                        combined_masks = [torch.cat(combined_masks)[sampled_indices]]
                        combined_labels = [torch.cat(combined_labels)[sampled_indices]]
                        if len(combined_tweet_ids):
                            combined_tweet_ids = [torch.cat(combined_tweet_ids)[sampled_indices]]

                        total_batch_selected_size = len(total_batch_selected_sample_indices[0])
                        is_done_selection = True
                        continue

                    if is_done_selection:
                        continue

                    if self.use_weak_label_info:
                        if self.use_sentence_transformer:
                            batch_inputs, batch_masks, batch_labels, batch_infos, batch_tweet_ids = batch
                        else:
                            batch_inputs, batch_masks, batch_labels, batch_infos = batch
                            batch_tweet_ids = None
                    else:
                        batch_inputs, batch_masks, batch_labels = batch
                        batch_infos = None
                    if device.type == 'cuda':
                        batch_inputs = batch_inputs.to(device)
                        batch_masks = batch_masks.to(device)
                        batch_labels = batch_labels.to(device)
                        if self.use_weak_label_info:
                            batch_infos = batch_infos.to(device)

                    if no_selector:
                        batch_random_selected_size = random.randrange(len(batch_labels))
                        batch_selected_sample_indices = random.sample(
                            range(len(batch_labels)), batch_random_selected_size)
                    else:
                        # Must forward pass the selector before get new selected indices
                        # Half of bags, get selected samples using the target selector
                        if idx % 2 == 0 or use_only_target_selector_to_select:
                            with torch.no_grad():
                                _ = self.reinforced_selector(
                                        batch_inputs, batch_masks, batch_labels,
                                        infos_from_weak_labeler=batch_infos,
                                        decide_actions=True, tweet_ids=batch_tweet_ids,
                                        device=device)
                                batch_selected_sample_indices = self.reinforced_selector.get_selected_sample_indices()

                        # The other half of bags, get selected samples using the current selector
                        else:
                            with torch.no_grad():
                                _ = tmp_reinforced_selector(
                                        batch_inputs, batch_masks, batch_labels,
                                        infos_from_weak_labeler=batch_infos,
                                        decide_actions=True, tweet_ids=batch_tweet_ids,
                                        device=device)
                                batch_selected_sample_indices = tmp_reinforced_selector.get_selected_sample_indices()

                    batch_selected_size = len(batch_selected_sample_indices)

                    if batch_selected_size > 0:
                        selected_train_inputs_weak = batch_inputs[batch_selected_sample_indices].cpu()
                        selected_train_masks_weak = batch_masks[batch_selected_sample_indices].cpu()
                        selected_train_labels_weak = batch_labels[batch_selected_sample_indices].cpu()
                        selected_train_tweet_ids_weak = batch_tweet_ids[batch_selected_sample_indices].cpu()

                        combined_inputs.append(selected_train_inputs_weak)
                        combined_masks.append(selected_train_masks_weak)
                        combined_labels.append(selected_train_labels_weak)
                        combined_tweet_ids.append(selected_train_tweet_ids_weak)

                        total_batch_selected_size += batch_selected_size
                        total_batch_selected_sample_indices.append(batch_selected_sample_indices)

                # Free space
                del batch_inputs
                del batch_masks
                del batch_labels
                del batch_infos

            if not fix_detector:
                # Combine manually-labeled samples
                for batch_inputs, batch_masks, batch_labels in train_manual_loader:
                    combined_inputs.append(batch_inputs)
                    combined_masks.append(batch_masks)
                    combined_labels.append(batch_labels)

                # Build combined data loader
                combined_dataset = TensorDataset(
                    torch.cat(combined_inputs),
                    torch.cat(combined_masks),
                    torch.cat(combined_labels)
                )
                combined_dataloader = DataLoader(
                    dataset=combined_dataset, batch_size=batch_size, shuffle=True)

                # To generate different results every epoch
                # set_seed(epoch * self.random_seed)

                # Train target fake news detector
                # Use the target detector as the main model after train on combined data
                results_epoch = train_model(
                    self.fake_news_detector, combined_dataloader,
                    validate_loader, detection_train_epoch, self.optimizer_fake_news_detector,
                    self.fake_news_detector.criterion, load_best_model=False,
                    early_stop=early_stop, device=device, eval_pos_label_only=False,
                    best_model_measure="f1", all_possible_labels=POSSIBLE_LABELS,
                    verbose=False, minority_class=self.minority_class)

                # Evaluate on test data every epoch
                if test_loader and isinstance(test_loader, DataLoader):
                    results_epoch_test = evaluate_model(
                        self.fake_news_detector, test_loader,
                        self.fake_news_detector.criterion, device=device,
                        eval_pos_label_only=False, verbose=False,
                        minority_class=self.minority_class)
                    # Combine to the result of this epoch to save the history performance
                    for metric in ["precision", "recall", "f1"]:
                        for c in POSSIBLE_LABELS:
                            results_epoch[c][f"test_{metric}"] = results_epoch_test[c][metric]
                    for metric in "loss", "accuracy":
                        results_epoch[f"test_{metric}"] = results_epoch_test[metric]

                # Save results every epoch but not save model
                dict_performance = {
                    "epoch": epoch,
                    "bag": bag_counter,
                    "baseline_performance": baseline_performance,
                    "total_epoch_reward": total_epoch_reward,
                    "total_batch_selected_size": total_batch_selected_size,
                    "results_epoch": results_epoch
                }

                # Save best epoch validation f1 fake news model
                # Consider class label 1
                if results_epoch[self.minority_class]["val_f1"] > best_epoch_val_f1:
                    best_epoch_val_f1 = results_epoch[self.minority_class]["val_f1"]
                    logger.info(f"Saving best epoch validation F1 model -> F1:{best_epoch_val_f1}")
                    tmp_prefix = checkpoint_prefix + "_best_epoch_validation_f1_"
                    self.best_epoch_val_f1_models_dir = self.store_best_models(
                        output_dir=output_dir, dict_performance=dict_performance, prefix=tmp_prefix)

            # Restore fake news detector in selector after `save_models`
            self.reinforced_selector.fake_news_detector = self.fake_news_detector

        return self.reinforced_selector, self.fake_news_detector


    @torch.no_grad()
    def update_target_policy_network(self, current_reinforced_selector_params,
                                     target_detector_params, learning_rate, device):
        # Selector
        for name, target_weight in self.reinforced_selector.named_parameters():
            current_weight = current_reinforced_selector_params[name].to(device)
            new_weight = DeMis.compute_target_policy_weight(
                target_weight, current_weight, learning_rate)
            target_weight.copy_(new_weight)

        # Detector
        if target_detector_params is not None:
            for name, current_weight in self.fake_news_detector.named_parameters():
                target_weight = target_detector_params[name].to(device)
                new_weight = DeMis.compute_target_policy_weight(
                    target_weight, current_weight, learning_rate)
                current_weight.copy_(new_weight)


    @torch.no_grad()
    def compute_target_policy_weight(target_weight, current_weight, learning_rate=0.001):
        return (1 - learning_rate) * target_weight + learning_rate * current_weight


    def store_best_models(self, output_dir, dict_performance, prefix):
        output_dir = f"{output_dir}/{prefix}best_model" if prefix else f"{output_dir}/best_model"
        self.save_models(output_dir, save_optimizers=True)  # save the detector and selector
        with open(f"{output_dir}/performance.json", "w") as f:
            json.dump(dict_performance, f, indent=4)
        return output_dir


    def save_models(self, output_dir="./", save_optimizers=False):
        if output_dir.endswith("/"):
            output_dir = output_dir[:-1]
        os.makedirs(output_dir, exist_ok=True)

        torch.save(
            self.fake_news_detector.state_dict(),
            f"{output_dir}/fake_news_detector.pt")

        if save_optimizers:
            torch.save(self.optimizer_fake_news_detector.state_dict(),
                   f"{output_dir}/optimizer_fake_news_detector.pt")
            torch.save(self.optimizer_selector.state_dict(),
                   f"{output_dir}/optimizer_selector.pt")

        self.reinforced_selector.fake_news_detector = None  # to save space
        torch.save(
            self.reinforced_selector.state_dict(),
            f"{output_dir}/reinforced_selector.pt")


    def load_models(self, model_dir, load_optimizers=False, device=None):
        if model_dir.endswith("/"):
            model_dir = model_dir[:-1]

        # Load model params
        if torch.cuda.is_available():
            self.fake_news_detector.load_state_dict(
                torch.load(f"{model_dir}/fake_news_detector.pt"))
            self.reinforced_selector.load_state_dict(
                torch.load(f"{model_dir}/reinforced_selector.pt"), strict=False)
        else:
            self.fake_news_detector.load_state_dict(
                torch.load(
                    f"{model_dir}/fake_news_detector.pt", map_location=torch.device('cpu')))
            self.reinforced_selector.load_state_dict(
                torch.load(
                    f"{model_dir}/reinforced_selector.pt", map_location=torch.device('cpu')),
                strict=False)

        # Load to device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fake_news_detector = self.fake_news_detector.to(device)
        self.reinforced_selector = self.reinforced_selector.to(device)
        self.reinforced_selector.fake_news_detector = self.fake_news_detector

        # Initialize optimizer objects
        self.optimizer_fake_news_detector = torch.optim.AdamW(
            self.fake_news_detector.parameters(), lr=self.learning_rate)
        self.optimizer_selector = torch.optim.AdamW(
            self.reinforced_selector.parameters(), lr=self.learning_rate)
            # self.reinforced_selector.parameters(), lr=self.learning_rate)

        if load_optimizers:
            try:
                # Update optimizer params
                self.optimizer_fake_news_detector.load_state_dict(
                    torch.load(f"{model_dir}/optimizer_fake_news_detector.pt"))
                self.optimizer_selector.load_state_dict(
                    torch.load(f"{model_dir}/optimizer_selector.pt"))
            except ValueError as e:
                error_str = "Value error when load optimizer, use the default optimizer instead."
                logger.error(f"{error_str} The error is: {e}")


def main():
    model_path = "vinai/bertweet-covid19-base-cased"
    demis_model = DeMis(model_path)

    print("DeMis model was loaded successfully")
    print(demis_model)

if __name__ == "__main__":
    main()