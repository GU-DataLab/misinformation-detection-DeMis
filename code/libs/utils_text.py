#!/usr/bin/env python
# coding: utf-8

'''
@title: Utilities for Text
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: Utilities for working with text
'''

import re
import contractions
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


stop_words_l = stopwords.words('english')
def clean_text(x):
    x = str(x)
    try:
        x = contractions.fix(x)
    except IndexError:
        pass
    x = x.replace("@USER", "")
    x = x.replace("<em>URL Removed</em>", "")
    x = x.replace("“", "")
    x = x.replace("”", "")
    x = x.replace("\n", " ")
    return " ".join(re.sub(r'[^a-zA-Z0-9_]',' ',w).lower().strip() for w in x.split() if re.sub(r'[^a-zA-Z0-9_]',' ',w).lower() not in stop_words_l)


def generate_ngrams(s, n, is_tweet=False):
    tokens = []
    if is_tweet:
        tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        tokens = tokenizer.tokenize(s)
    else:
        tokens = word_tokenize(s)
    output = list(ngrams(tokens, n))
    return output


def censor_text(text):
    # Censor mentions and links
    text = re.sub(r"@[\w|\d|_]+[^ |@]", "@USER", text)
    text = re.sub(r"(http://[^ ]+)", "HTTPURL", text)
    text = re.sub(r"(https://[^ ]+)", "HTTPURL", text)
    return text


def remove_spaces(text):
    return re.sub(' +', ' ', text)


def main():
    s = """Natural-language processing (NLP) is or isn't an area of
    computer science and artificial intelligence
    concerned with the interactions between computers
    and human (natural) languages.
    """
    
    ngram = generate_ngrams(s, n=3)
    for n in ngram:
        print(n)

    sample_text = "@kornraphop Hello sir! Here is the link to the website https://ABC.com"
    print(censor_text(sample_text))


if __name__ == "__main__":
    main()