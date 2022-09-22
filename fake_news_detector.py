#!/usr/bin/env python
# coding: utf-8

'''
@title: Fake News Detector
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: Fake news detector class for fake news detection.
'''


import sys
import argparse

import torch
from torch._C import Value
import torch.nn as nn

from transformers import AdamW, AutoConfig

sys.path.append("../.")
from textual_feature_extractor import TextualFeatureExtractor
from libs.utils_misc import set_seed


class ClassificationHead(nn.Module):
    """Classification head for the RoBERTa-based models.

    Ref: https://huggingface.co/transformers/_modules/transformers/models/roberta/modeling_roberta.html#RobertaForSequenceClassification
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class FakeNewsDetectorTransformer(nn.Module):
    def __init__(self, model_path, loss_weights=None, num_labels=2):
        super(FakeNewsDetectorTransformer, self).__init__()

        if num_labels != 2:
            raise ValueError(f"Only works for binary class. Found `num_labels`: {num_labels}.")
        self.num_labels = num_labels

        # Prepare textual extractor
        if isinstance(model_path, str):
            self.extractor = TextualFeatureExtractor(model_path)
            self.config = AutoConfig.from_pretrained(model_path)
        elif isinstance(model_path, TextualFeatureExtractor):
            self.extractor = model_path
            self.config = self.extractor.config
        else:
            raise ValueError(f"Invalid type of `model_path`: {type(model_path)}")
        
        # Setup model config
        self.config.__dict__["num_labels"] = num_labels  # is myth or not myth
        self.config.__dict__["classifier_dropout"] = None

        # Criterion
        if loss_weights is not None:
            assert len(loss_weights) == num_labels
        self.criterion = nn.CrossEntropyLoss(weight=loss_weights)

        # Classifier
        self.classifier = ClassificationHead(self.config)

        # Init with positive weights so high chance to select
        # at the few first batches to avoid zero selected.
        for name, param in self.classifier.named_parameters():
            with torch.no_grad():
                if param.requires_grad and name.endswith(".bias"):
                    param.copy_(torch.zeros(param.size(), requires_grad=True))


    def forward(self, embedding_indices, mask):
        # embedding_indices, mask = input_tuple
        input_tuple = (embedding_indices, mask)
        outputs = self.extractor(input_tuple)  # embedding_indices, mask
        sequence_output = outputs[0]  # first element is `last_hidden_state`
        logits = self.classifier(sequence_output)
        return logits

    def get_loss(self, logits, labels):
        return self.criterion(logits.view(-1, self.num_labels), labels.view(-1))

    def get_feature_vectors(self, embedding_indices, mask) -> torch.Tensor:
        input_tuple = (embedding_indices, mask)
        outputs = self.extractor(input_tuple)
        return outputs

    def generate_emb_indices_and_mask(self, texts, padding="max_length",
                                      max_seq_length=128):
        return self.extractor.generate_emb_indices_and_mask(
            texts, padding=padding, max_seq_length=max_seq_length)

    def predict_probs(self, texts, device=None):
        embedding_indices, mask = self.generate_emb_indices_and_mask(texts)
        if device:
            embedding_indices = embedding_indices.to(device)
            mask = mask.to(device)
        logits = self.forward(embedding_indices, mask)
        fn_softmax = nn.Softmax(dim=1)
        return fn_softmax(logits)

    def predict(self, texts, device):
        probs = self.predict_probs(texts, device)
        return torch.argmax(probs, dim=1)


def parse_arguments(parser):
    parser.add_argument('--model_path', type=str, default="vinai/bertweet-covid19-base-cased")
    parser.add_argument('--input_filepath', type=str, default=None)
    return parser


def main(args):
    set_seed(k=3407)

    if args.input_filepath:
        raise NotImplementedError
    else:
        # Samples
        texts = ["This is True", "Yeah Yeah", "This is False"]

    fake_news_detector = FakeNewsDetectorTransformer(model_path=args.model_path)
    optimizer = AdamW(fake_news_detector.parameters(), lr=0.0001)
    embedding_indices, mask = fake_news_detector.generate_emb_indices_and_mask(texts)
    labels = torch.tensor([1, 0, 0])

    if torch.cuda.is_available():
        fake_news_detector.cuda()
        embedding_indices = embedding_indices.to("cuda")
        mask = mask.to("cuda")
        labels = labels.to("cuda")

    print("Feature vectors")
    print(fake_news_detector.get_feature_vectors(embedding_indices, mask))
    # input_tuple = (embedding_indices, mask)
    logits = fake_news_detector(embedding_indices, mask)
    print("Output")
    print(logits)
    print("View")
    print(logits.view(-1, 2))
    # print("Detector")
    # print(fake_news_detector)
    print("Loss")
    loss = fake_news_detector.get_loss(logits, labels=labels)
    print(loss)

    print("A layer BEFORE backward")
    print(fake_news_detector.classifier.out_proj.weight)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("A layer AFTER backward")
    print(fake_news_detector.classifier.out_proj.weight)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    args = parser.parse_args()
    main(args)