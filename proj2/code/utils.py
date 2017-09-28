import numpy as np
import pickle

dtype = np.float32

def read_file_by_line(data_path):
    # read the given file line by line
    f = open(data_path,'r')
    data_list = []
    while True:
        line = f.readline()
        if line == '':
            break
        data_list.append(line)
    f.close()
    return data_list

def cat_sent(sents):
    new_sents = []
    for sent in sents:
        sent = '<s>'+' '+sent+' '+'</s>'
        new_sents.append(sent)
    return new_sents

def list_to_dict(token_list):
    token_dict = {}
    for idx, token in enumerate(token_list):
        token_dict[token] = idx
    return token_dict

def get_context_mat(sents, vocab_dict, context_dict, w):
    # the function creates the matrix of context word counts
    V, V_C = len(vocab_dict), len(context_dict)
    word_emb = np.zeros((V, V_C), dtype=dtype)
    for sent in sents:
        # go through all the sentences
        tokens_sent = sent.split()
        len_sent = len(tokens_sent)
        for token_id, token in enumerate(tokens_sent):
            if token in vocab_dict:
                # if the token is in vocaburaly
                row_id = vocab_dict[token]
                # check the +/- w tokens around this token
                start_idx = token_id - w  # 0,1,2,3  4, 0 is the start_idx
                end_idx = token_id + w + 1 # 4   5,6,7,8  9, 9 is the end_idx
                if start_idx < 0:
                    start_dix = 0
                if end_idx > len_sent:
                    end_dix = len_sent
                # get the 1st side
                for idx in range(start_idx, token_id):
                    context_word = tokens_sent[idx]
                    if context_word in context_dict:
                        col_id = context_dict[context_word]
                        word_emb[row_id, col_id] += 1.0
                # get the 2nd side
                for idx in range(token_id+1, end_idx):
                    context_word = tokens_sent[idx]
                    if context_word in context_dict:
                        col_id = context_dict[context_word]
                        word_emb[row_id, col_id] += 1.0
    return word_emb

def get_ppmi(sents, vocab_dict, context_dict, w):
    # the function creates the matrix of ppmi
    word_emb = get_context_mat(sents, vocab_dict, context_dict, w)
    freq_mat = word_emb / np.sum(word_emb)
    freq_i_row = np.sum(word_emb, axis=1) / np.sum(word_emb)
    freq_j_col = np.sum(word_emb, axis=0) / np.sum(word_emb)
    base_mat = freq_i_row[:,None] * freq_j_col[None,:]
    word_emb = word_emb / base_mat
    eps = np.float32(1e-6)
    word_emb[(word_emb <= eps)] = 0.0
    return word_emb

def get_data(corpus_path, vocab_path, context_path, mode, w):
    # the function creates the matrix of |V|*|V_C|
    assert ((mode=='word-context') or (mode=='ppmi'))
    #
    sents = read_file_by_line(corpus_path)
    sents = cat_sent(sents)
    vocab_list = read_file_by_line(vocab_path)
    context_list = read_file_by_line(context_path)
    print sents[-1], vocab_list[-1], context_list[-1]
    #
    vocab_dict = list_to_dict(vocab_list)
    context_dict = list_to_dict(context_list)
    #
    if mode == 'word-context':
        word_emb = get_context_mat(sents, vocab_dict, context_dict, w)
    elif mode == 'ppmi':
        word_emb = get_ppmi(sents, vocab_dict, context_dict, w)
    else:
        print "no such mode found"
    return word_emb
