import numpy as np
from utils import *
import pickle

#TODO: read data
corpus_path = '../data/wiki-0.1percent.txt'
vocab_path = '../data/vocab-15k.txt'
context_path = '../data/vocab-10k.txt'

mode = 'word-context'
# mode can be 'word-context' or 'ppmi'
w = 4

word_emb = get_data(
    corpus_path, vocab_path, context_path, mode, w)

#TODO: define the model and train

np.random.seed(12345)
