#!/usr/bin/env python
# coding: utf-8

'''
@title: Myth Tweet Data
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: Tweet data for misinformation detection.
'''


import pandas as pd
import collections

from torch.utils.data import Dataset, DataLoader


class MythTweetData(Dataset):
    STATE_INFO_NAME_TO_ID = {
        "sim_top_1th": 0,
        "sim_top_kth": 1,
        "sim_highest_target_claim": 2
    }

    def __init__(self, df, use_weak_label_info=False, use_tweet_ids=False):
        self.use_weak_label_info = use_weak_label_info
        self.use_tweet_ids = use_tweet_ids
        self.texts = list(df["text"])
        self.labels = list(df["label"]) if "label" in df.columns else list(df["weak_label"])
        # print(f"Data Size: {len(self.labels)}")
        if len(self.texts) != len(self.labels):
            raise ValueError(f"Size of texts and labels are not similar.")

        self.label_counter = collections.Counter(self.labels)

        # Weak label info
        if self.use_weak_label_info:
            self.weak_label_info = [
                [0.0]*len(MythTweetData.STATE_INFO_NAME_TO_ID)
                for _ in range(len(self.labels))]
            for idx, row in df.iterrows():
                for state_info_name, state_info_idx in MythTweetData.STATE_INFO_NAME_TO_ID.items():
                    self.weak_label_info[idx][state_info_idx] = row[state_info_name]

        # Tweet IDs
        if self.use_tweet_ids:
            self.tweet_ids = list(df["tweet_id"])
            # Validate
            for t in self.tweet_ids:
                if not isinstance(t, str):
                    raise ValueError(f"Tweet ID should be read as string. "
                                     f"Found {t} of type {type(t)}")

    def __len__(self):
        return len(self.labels)

    def get_minority_class(self):
        # NOTE: Only works for binary class
        print(self.label_counter)
        return self.label_counter.most_common(2)[-1][0]

    def get_class_size(self, class_label):
        """ Get number of instances of the class. """
        return self.label_counter[class_label]

    def __getitem__(self, idx):
        if self.use_weak_label_info:
            if self.use_tweet_ids:
                return self.texts[idx], self.labels[idx], self.weak_label_info[idx], self.tweet_ids[idx]
            else:
                return self.texts[idx], self.labels[idx], self.weak_label_info[idx]
        else:
            if self.use_tweet_ids:
                raise ValueError("If NOT use_weak_label_info, then cannot use_tweet_ids")
            return self.texts[idx], self.labels[idx]


class ReadyData(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def undersampling(df, label_col, random_state=3407):
    count_class = collections.Counter(df[label_col].tolist())
    min_count = min([v for _, v in count_class.items()])

    dfs = []
    for c in count_class:
        df_tmp = df[df[label_col]==c]
        df_tmp = df_tmp.sample(n=min_count, random_state=random_state)
        dfs.append(df_tmp)
    df_balanced = pd.concat(dfs, ignore_index=True)
    return df_balanced


def oversampling(df, label_col, random_state=3407):
    count_class = collections.Counter(df[label_col].tolist())
    max_count = max([v for _, v in count_class.items()])

    dfs = []
    for c in count_class:
        df_tmp_c = df[df[label_col]==c]
        # Duplicate minority class samples
        multiply_num = int(max_count / df_tmp_c.shape[0])
        df_mul = pd.concat([df_tmp_c] * multiply_num, ignore_index=True)
        # Fill the rest
        left_num = max_count % df_tmp_c.shape[0]
        df_tmp_more = df_tmp_c.sample(n=left_num, random_state=random_state)
        dfs.append(pd.concat([df_mul, df_tmp_more], ignore_index=True))
    df_balanced = pd.concat(dfs, ignore_index=True)
    return df_balanced


def main():
    filepath = "../../data/twitter/covid-19/home-remedies-779/train-val-test/home_remedies_myth_test_more_info.csv"
    df = pd.read_csv(filepath)

    # Do not use weak label information
    dataset = MythTweetData(df)

    print("Check loading data")
    for idx, (text, label) in enumerate(dataset):
        if idx == 5:
            break
        print(f"{label}: {text}")

    print("Check using batch")
    data_loader = DataLoader(dataset=dataset,
                              batch_size=3,
                              shuffle=True)

    for i, (texts, labels) in enumerate(data_loader):
        print(f"Batch: {i}")
        print(texts)
        print(labels)
        if i == 5:
            break

    # Use information from weak labeler
    dataset = MythTweetData(df, use_weak_label_info=True)

    print("Check loading data")
    for idx, (text, label, info) in enumerate(dataset):
        if idx == 5:
            break
        print(f"{label}: {text}, {info}")

    print("Check using batch")
    data_loader = DataLoader(dataset=dataset,
                              batch_size=3,
                              shuffle=True)

    for i, (texts, labels, infos) in enumerate(data_loader):
        print(f"Batch: {i}")
        print(texts)
        print(labels)
        print(infos)
        print("One value")
        print(infos[0][0].item(), infos[0][1], infos[0][2])
        if i == 5:
            break

    # Try oversampling
    print(f"Imbalance: {collections.Counter(df['label'])}")
    df_balance = oversampling(df, "label")
    print(f"Balance: {collections.Counter(df_balance['label'])}")


if __name__ == "__main__":
    main()