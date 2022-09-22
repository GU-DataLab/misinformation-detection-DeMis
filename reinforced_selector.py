#!/usr/bin/env python
# coding: utf-8

'''
@title: Reinforced Selector
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: Data selector using reinforcement learning for DeMis.
'''


import sys
import time
import copy
import numpy as np

import torch
from torch.autograd.grad_mode import no_grad
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append("../.")
from libs.utils_misc import set_seed
from libs.utils_data import MythTweetData
from fake_news_detector import FakeNewsDetectorTransformer
from sim_sentence_embedding import mean_pooling


STATE_SIZE = {
    "with_info": 8,
    "without_info": 3
}


@torch.no_grad()
def get_action(prob, use_random=True, threshold=0.5):
    if use_random:
        t = 1000 * time.time() # current time in milliseconds
        set_seed(int(t) % 2**32)  # seed must be between 0 to 2^32 - 1
        random_prob = np.random.rand()
        return (prob > random_prob) * 1
    else:
        return (prob > threshold) * 1


@torch.no_grad()
def get_cosine_similarity(input1, input2):
    cos = nn.CosineSimilarity(dim=0, eps=1e-08)
    return cos(input1, input2)


class Environment(object):
    def __init__(self):
        super(Environment, self).__init__()

    @torch.no_grad()
    def reset(self, batch_sentence_ebd, batch_mask, batch_label,
              probs_from_detector, info_from_weak_labeler=None,
              state_size=3):
        self.state_size = state_size
        self.batch_len = len(batch_sentence_ebd)
        self.sentence_ebd = batch_sentence_ebd
        self.batch_mask = batch_mask
        self.labels = batch_label
        self.probs_from_detector = probs_from_detector
        self.info_from_weak_labeler = info_from_weak_labeler
        self.current_step = 0
        self.num_selected = 0
        self.list_selected = []
        self.list_action_prob = []

        # Elements in a current state
        self.vector_mean = np.array([0.0 for _ in range(self.state_size)], dtype=np.float32)
        self.vector_sum = np.array([0.0 for _ in range(self.state_size)], dtype=np.float32)
        max_cosine_similarity = 0.0

        # A state contains 3 or 8 elements (in/ex-cluding the prob from weak label)
        current_state = [
            float(self.probs_from_detector[self.current_step]),
            float(max_cosine_similarity),
            float(self.labels[self.current_step])]

        if self.info_from_weak_labeler is not None:
            state_info = self.build_info_state()
            current_state = state_info + current_state
            assert len(current_state) == STATE_SIZE["with_info"], "Please verify state."

        assert self.state_size == len(current_state), \
            f"Mismatched `state_size`: {self.state_size} but " \
            f"found {len(current_state)}"

        self.final_state = current_state + self.vector_mean.tolist()  # concat
        return self.final_state

    @torch.no_grad()
    def step(self, action):
        if action == 1:
            self.num_selected += 1
            self.list_selected.append(self.current_step)

        self.vector_sum = self.vector_sum + (action * np.array(self.final_state[:self.state_size], dtype=np.float32))
        if self.num_selected > 0:
            self.vector_mean = self.vector_sum / self.num_selected

        # Update to next step
        self.current_step += 1
        if self.current_step < self.batch_len:  # do not compute more than batch size
            max_cosine_similarity = self.get_current_max_cosine_sim()
            current_state = [
                float(self.probs_from_detector[self.current_step]),
                float(max_cosine_similarity),
                float(self.labels[self.current_step])]
            if self.info_from_weak_labeler is not None:
                state_info = self.build_info_state()
                current_state = state_info + current_state
                assert len(current_state) == STATE_SIZE["with_info"], "Please verify state."
            self.final_state = current_state + self.vector_mean.tolist()  # concat

        return self.final_state

    @torch.no_grad()
    def get_current_max_cosine_sim(self):
        max_sim = 0
        for idx in self.list_selected:
            # Do not compute similarity to itself
            if idx == self.current_step:
                continue

            current_ebd = self.sentence_ebd[self.current_step]
            a_ebd = self.sentence_ebd[idx]
            sim = get_cosine_similarity(current_ebd, a_ebd).item()
            max_sim = max(max_sim, sim)
        return max_sim

    @torch.no_grad()
    def build_info_state(self):
        # Similarity to the top-1-th claim
        sim_top_1th = self.info_from_weak_labeler[self.current_step][
            MythTweetData.STATE_INFO_NAME_TO_ID["sim_top_1th"]].item()

        # Similarity to the top-k-th claim
        sim_top_kth = self.info_from_weak_labeler[self.current_step][
            MythTweetData.STATE_INFO_NAME_TO_ID["sim_top_kth"]].item()

        # Similarity to the highest matched target claim
        sim_highest_target_claim = self.info_from_weak_labeler[self.current_step][
            MythTweetData.STATE_INFO_NAME_TO_ID["sim_highest_target_claim"]].item()

        # Difference of similarity score between the highest target
        # claims and the top 1-th claim
        diff_sim_1 = sim_top_1th - sim_highest_target_claim

        # Difference of similarity score between the highest target
        # claims and the top k-th claim
        diff_sim_2 = sim_highest_target_claim - sim_top_kth

        return [sim_top_1th, sim_top_kth, sim_highest_target_claim,
                diff_sim_1, diff_sim_2]


class ReinforcedSelector(nn.Module):
    # Load once
    global_sentence_embeddings = None  # tweet IDs to sentence embeddings


    def load_global_sentence_embeddings(filepaths, device=None):
        """Load a dict of tweet ID to sentence embeddings."""
        if ReinforcedSelector.global_sentence_embeddings is not None:
            raise ValueError("`global_sentence_embeddings` is already loaded.")

        if isinstance(filepaths, str):
            filepaths = [filepaths]

        tmp_dict = {}
        for fp in filepaths:
            tmp = torch.load(fp) if device and device.type == "cuda" else torch.load(fp, map_location=torch.device('cpu'))
            tmp_dict = {**tmp_dict, **tmp}  # merge dicts
        ReinforcedSelector.global_sentence_embeddings = copy.copy(tmp_dict)
        del tmp_dict


    def _get_sentence_embeddings(tweet_ids: torch.LongTensor):
        list_embeddings = [
            ReinforcedSelector.global_sentence_embeddings[str(tweet_id.item())]
            for tweet_id in tweet_ids
        ]
        return torch.stack(list_embeddings)


    def __init__(self, fake_news_detector, hidden_size=8, state_size=3):
        super(ReinforcedSelector, self).__init__()
        self.fake_news_detector = fake_news_detector
        self.hidden_size = hidden_size
        self.state_size = state_size

        # Selector is a 2-layer of NN with ReLU and Sigmoid
        self.selector = nn.Sequential()
        self.selector.add_module('nn1', nn.Linear(self.state_size * 2, self.hidden_size))
        self.selector.add_module('relu', nn.ReLU())
        self.selector.add_module('nn2', nn.Linear(self.hidden_size, 1))
        self.selector.add_module('sigmoid', nn.Sigmoid())

        # Init with positive weights so high chance to select
        # at the few first batches to avoid zero selected.
        for name, param in self.selector.named_parameters():
            with torch.no_grad():
                if param.requires_grad and name.endswith(".bias"):
                    param.copy_(torch.zeros(param.size(), requires_grad=True))


    def forward(self, embedding_indices, mask, labels,
                infos_from_weak_labeler=None, decide_actions=False,
                tweet_ids=None, device="cpu"):
        """
        Args:
            info_from_weak_labeler (Tensor): Information from weak labeler. This
                will be used to form RL states.
            tweet_ids (list): A list of tweet IDs used to query loaded sentence embeddings.
        """

        if infos_from_weak_labeler is not None:
            infos_from_weak_labeler = infos_from_weak_labeler.detach().cpu()

        assert len(embedding_indices) == len(labels)
        if not torch.is_tensor(embedding_indices):
            raise ValueError(f"Weird type of embedding_indices: {type(embedding_indices)}")

        # From textual extractor
        # During RL training, we won't backprop the textual extractor
        with torch.no_grad():
            if tweet_ids is not None:
                assert isinstance(tweet_ids, list) or isinstance(tweet_ids, torch.LongTensor)
                assert len(tweet_ids) == len(labels)
                batch_sentence_ebd = ReinforcedSelector._get_sentence_embeddings(tweet_ids)
                batch_sentence_ebd = batch_sentence_ebd.to(device)
            else:
                batch_outputs = self.fake_news_detector.get_feature_vectors(
                    embedding_indices, mask)
                batch_sentence_ebd = mean_pooling(batch_outputs, mask)  # from word to sentence embeddings
                batch_sentence_ebd = batch_sentence_ebd.detach().clone()

        # Get prob of being class 1 (being myth) from detector
        with torch.no_grad():
            probs_from_detector = self.fake_news_detector(embedding_indices, mask)
        # Probs of output class 1
        probs_from_detector = probs_from_detector.transpose(0, 1).detach().clone()[1]

        # Init state
        self.env = Environment()
        state = self.env.reset(batch_sentence_ebd, mask, labels, probs_from_detector,
                               infos_from_weak_labeler, self.state_size)

        self.list_actions = []
        list_states = []

        # Process RL step
        with torch.no_grad():
            for i in range(len(labels)):
                state = torch.tensor([state], device=device)
                prob = self.selector(state)
                prob = prob.detach().clone()
                list_states.append(state)
                self.env.list_action_prob.append(prob[0][0].item())

                # Extract policy value
                if decide_actions:
                    action = 0 if prob[0][0].item() < 0.5 else 1
                else:
                    action = int(torch.sum(get_action(prob)))
                self.list_actions.append(action)

                if action not in [0, 1]:
                    raise ValueError(f"action here must be int of 1 or 2. Found {action}.")

                # Policy is dynamically changed based on action
                if torch.is_tensor(action):
                    raise NotImplementedError(
                        f"action in reinforced selector must be type of "
                        f"int. Found {type(action)}")

                # Update state
                state = self.env.step(action)

        # Once we get all the actions and state based on sampling
        # Now we compute actual actions based on the states we pre-compute
        self.final_prob = self.selector(torch.cat(list_states))

        # Compute policy using the final probs
        actions = torch.tensor(self.list_actions, device=device, requires_grad=False)\
            .reshape(tuple(self.final_prob.size()))
        policy = actions * self.final_prob + (1 - actions) * (1 - self.final_prob)

        return policy


    def get_selected_sample_indices(self):
        return self.env.list_selected

    def get_action_probs(self):
        return self.env.list_action_prob


def main():
    set_seed(3407)

    # Test
    detector = FakeNewsDetectorTransformer(model_path="vinai/bertweet-covid19-base-cased")
    selector = ReinforcedSelector(detector)

    texts = ("I love you", "I am Ken")
    labels = torch.tensor([1, 0])

    embedding_indices, mask = selector.fake_news_detector.generate_emb_indices_and_mask(texts)

    policy_values = selector(embedding_indices, mask, labels)
    print(f"Text: {texts}")
    print(f"Labels: {labels}")
    print(f"Policy: {policy_values}")
    print(f"Sampled Action: {get_action(torch.flatten(policy_values))}")

    policy_values = selector(embedding_indices, mask, labels, decide_actions=True)
    print(f"Decided Policy: {policy_values}")
    print(f"Decided Action: {selector.get_selected_sample_indices()}")

    # Test `get_cosine_similarity`
    A = np.array([[10.0, 3.0], [11.0, 4.0]])
    B = np.array([[8.0, 7.0], [9.0, 8.0]])
    result = cosine_similarity(A.reshape(1,-1),B.reshape(1,-1))
    print(A)
    print(B)
    print(result)
    output = get_cosine_similarity(torch.tensor(A), torch.tensor(B))
    print(output)

if __name__ == "__main__":
    main()