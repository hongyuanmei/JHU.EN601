
# coding: utf-8

# In[178]:

import numpy as np
import nltk
import time
from sklearn.metrics.pairwise import cosine_similarity


# In[179]:

def getMap(word):
    w = {}
    for (n,i) in enumerate(word):
        i = i.strip()
        w[i] = n
    return w


# In[180]:

def get_index(word_map,w):
    if w in word_map:
        return word_map[w]
    else:
        return None


# In[181]:

def get_matrix(filename):
    start_time = time.time()
    c = np.zeros((len(w),len(wc)))
    lines = open(filename,'r').readlines()
    w_size = 4
    for line in lines:
        word_list = nltk.word_tokenize(line)
        word_list.insert(0, '<s>')
        word_list.append('</s>')
        for i in range(len(word_list)):
            c_index = get_index(w,word_list[i])
            if c_index != None:
                for j in range(1,w_size+1):
                    if i-j >= 0:
                        be_index = get_index(wc,word_list[i-j])
                        if be_index != None:
                            c[c_index][be_index] += 1
                    if i+j < len(word_list):
                        for_index = get_index(wc,word_list[i+j])
                        if for_index != None:
                            c[c_index][for_index] += 1
    end_time = time.time()
    print "Total time:", (end_time - start_time)
    return c


# In[182]:

def eva(c,qword):
    targ_index = get_index(w,qword)
    result = (cosine_similarity(c[targ_index], c)[0])
    index_list = [i[0] for i in sorted(enumerate(result), key=lambda x:x[1],reverse=True)[:6]][1:]
    print ''
    print ''
    print 'Nearest neighbor for ',qword
    for i in index_list:
        print word[i].strip(),


# In[ ]:

word = open('../data/vocab-15k.txt','r').readlines()
wordc = open('../data/vocab-10k.txt','r').readlines()
w = getMap(word)
wc = getMap(wordc)
c = get_matrix('../data/wiki-0.1percent.txt')

t1 = time.time()
eva_list = ['people', 'flew', 'transported', 'quickly', 'good', 'python', 'apple', 'red', 'chicago', 'language']
for qword in eva_list:
    eva(c,qword)
t2 = time.time()
print 'Evaluation Time',(t2-t1)


# In[ ]:
