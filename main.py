import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from itertools import groupby
from operator import itemgetter


# import numpy as np
# import pandas as pd
# import os
# import codecs
# from sklearn import feature_extraction
# import mpld3


def get_tokens(text):
    lowers = text.lower()
    exclude = set(string.punctuation)
    no_punctuation = ''.join(word for word in lowers if word not in exclude)

    tokens = nltk.word_tokenize(no_punctuation)
    return tokens


def stopword_removal_english(tokens):
    stop_word = set(stopwords.words('english'))
    return [w for w in tokens if w not in stop_word]


def stopword_removal_indonesia(tokens):
    stop_word = set(stopwords.words('indonesia'))
    return [w for w in tokens if w not in stop_word]


def stemmer_english(tokens):
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(t) for t in tokens]


def stemmer_indonesia(tokens):
    stemmer = StemmerFactory().create_stemmer()
    return [stemmer.stem(t) for t in tokens]


if __name__ == '__main__':
    file_eng = "data-english.txt"
    file_ind = "data-indo.txt"

    with open(file_eng, 'r') as file:
        text = file.read()
        text = get_tokens(text)
        text = stopword_removal_english(text)
        text = stemmer_english(text)

        tfidf_vectorizer = TfidfVectorizer(stop_words='english', tokenizer=get_tokens)
        tfidf_matrix = tfidf_vectorizer.fit_transform(text)

        num_cluster = 5
        km = KMeans(n_clusters=num_cluster)
        km.fit(tfidf_matrix)
        clusters = km.labels_.tolist()

        zip_list = list(zip(clusters, text))
        print(zip_list)

    print("\n")

    with open(file_ind, 'r') as file:
        text = file.read()
        text = get_tokens(text)
        text = stopword_removal_indonesia(text)
        text = stemmer_indonesia(text)

        tfidf_vectorizer = TfidfVectorizer(tokenizer=get_tokens)
        tfidf_matrix = tfidf_vectorizer.fit_transform(text)

        km = KMeans(n_clusters=5)
        km.fit(tfidf_matrix)
        clusters = km.labels_.tolist()

        zip_list = list(zip(clusters, text))
        print(zip_list)
