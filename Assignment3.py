import pandas as pd
import numpy as np 
from pandas import read_csv
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def softmax(x):
    e = np.exp(x)
    return e/np.sum(e)

data = read_csv('rnnDataset.csv')

t = []
all_t = []
vocab = []

stop_words = set(stopwords.words('english'))

for x in range(len(data)):
    ts = ["START"]
    for tk in word_tokenize(data.loc[x][0]):
        tl = tk.lower()
        if(tl not in stop_words):
            ts.append(tk)
            all_t.append(tk)
    ts.append("END")
    t.append(ts)

fdist = nltk.FreqDist(all_t)

for w, frequency in fdist.most_common(8000):
    vocab.append(w)

vocab.append("START")
vocab.append("END")
vocab.append("UNK")

for x in range(len(t)):
    for y in range(len(t[x])):
        if(t[x][y] not in vocab):
            t[x][y] = "UNK"

input_dim  = 10
hidden_dim = 100
output_dim = 10

tt = Word2Vec(t, size = input_dim)

# for x in range(10):
#     s = t[x]
#     print(s)
#     print(tt[s])

U = np.random.uniform(0,1, (hidden_dim, input_dim))
W = np.random.uniform(0,1, (hidden_dim, hidden_dim))
V = np.random.uniform(0,1, (output_dim, hidden_dim))
prev_states = []

for i in range(len(t)):
    s = t[i]
    v = tt[s]
    prev_s = np.zeros((hidden_dim,1))
    count = 0
    for wv in v:
        count += 1
        current_hidden = np.dot(U,wv)
        previous_hidden = np.dot(W, prev_s)
        added = current_hidden + previous_hidden
        state = np.tanh(added)
        prev_s = state
        if(len(t)-count < 6):
            prev_states.append(state)
        output = np.dot(V, state)
        output = softmax(output)
        print(output)