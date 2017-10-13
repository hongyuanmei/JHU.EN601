import numpy as np
from utils import *
import pickle

np.random.seed(12345)

#TODO: read data
#corpus_path = '../data/wiki-0.1percent.txt'
#vocab_path = '../data/vocab-15k.txt'
#context_path = '../data/vocab-10k.txt'
#
corpus_path = '../data/learnprogramming.txt'
vocab_path = '../data/learnprogramming.vocab'
context_path = '../data/learnprogramming.vocab'

mode = 'word-context'
# mode can be 'word-context' or 'ppmi'
w = 4

word_emb, vocab_dict, context_dict = get_data(
    corpus_path, vocab_path, context_path, mode, w)

print vocab_dict.keys()
print sum(word_emb)

#TODO: define the model and train

candidates = [
    #'leetcode',
    'python', 'function', 'threads', 'variables', 'hash'
]

neighbors = get_neighbors(
    candidates, 5, word_emb, vocab_dict
)

print neighbors
