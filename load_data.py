# coding: utf-8

import pandas as pd
import ast
import json
import subprocess as sb
import numpy as np


data = "data.json"
data = "just_restaurants.json"
with open(data, 'r') as r:
    i = 0
    lines = []
    for x in r:
        lines.append(pd.DataFrame([json.loads(x)]))
        i += 1
        if i % 10000 == 0:
            print(i)
    res = pd.concat(lines)


def get_reviews(like_type):
    n = res[res['business_review_count'] > 50].count()[0]
    # print(n)
    # print(type(n))
    tmp = res[res['business_review_count'] > 50].sample(n=n).copy()
    n = tmp[tmp['attributes_count'] > 10].count()[0]
    rs = tmp[tmp['attributes_count'] > 10].sample(n=n).copy()

    pos = rs[rs[like_type] > 2].sample(n=10000).copy()
    pos['classification'] = like_type
    neg = rs[rs[like_type] == 0].sample(n=10000).copy()
    neg['classification'] = 'not-' + like_type
    all = pd.concat([pos, neg])
    all = all[['text', like_type, 'classification', 'stars', 'business_id', 'words', 'incorrect_words']].reset_index(
        drop=True)
    return all


# # Classification



import nltk

toker = nltk.tokenize.TweetTokenizer()

# In[23]:


like_type = 'useful'
# like_type='funny'
# like_type='cool'


# In[24]:


reviews = get_reviews(like_type)

# In[25]:


texts_tokenized = (toker.tokenize(row.text) for index, row in reviews.iterrows())
all_words = nltk.FreqDist(w.lower() for tokens in texts_tokenized for w in tokens)

print('total number of words:', sum(all_words.values()))
print('unique words:', len(all_words))
print('words present only once:', sum(c for c in all_words.values() if c == 1))

all_words.plot(30)

# In[26]:


words = all_words.copy()
for w, count in all_words.items():
    if count > 1000 or count == 1:
        del words[w]

print('feature words:', len(words))
words.plot(40)

# In[27]:


top_words = words.copy()
for w, count in all_words.items():
    if count > 200 or count <= 20:
        del top_words[w]

print('feature words:', len(top_words))
top_words.plot(40)
top_words = frozenset(top_words.keys())

# In[28]:


word_features = frozenset(words.keys())
i = 0
words_numbered = dict()
for w in word_features:
    words_numbered[w] = i
    i += 1

# In[29]:


len(word_features)


# In[30]:


def doc2vec(text):
    return [(i, words_numbered[w]) for i, w in enumerate(toker.tokenize(text.lower())) if w in words_numbered]


# In[31]:


import random

# In[32]:


# cosine similarity
corpus = [doc2vec(t) for t in random.sample(list(reviews[reviews['classification'] == 'useful']['text']), 10)]

# In[33]:


from gensim.similarities.docsim import Similarity
from gensim.test.utils import get_tmpfile

# In[34]:


index = Similarity(None, corpus, num_features=len(words_numbered))  # create index


# In[35]:


def features(row):
    text = row.text
    txt_words = set(toker.tokenize(text.lower()))
    features = {}

    for w in txt_words & top_words:
        features['contains({})'.format(w)] = 'Yes'  # beze slov je to lepsi
        pass

    for w in txt_words & word_features:
        # features['contains({})'.format(w)] = 'Yes' # beze slov je to lepsi
        pass

    text_tok = toker.tokenize(text.lower())
    for w, w2 in zip(text_tok[:-1], text_tok[1:]):
        if w in word_features and w2 in word_features:
            features['contains({}&&&{})'.format(w, w2)] = 'Yes'
            pass

    for (w, w2), w3 in zip(zip(text_tok[:-2], text_tok[1:-1]), text_tok[2:]):
        if w in word_features and w2 in word_features and w3 in word_features:
            features['contains({}&&&{}&&&{})'.format(w, w2, w3)] = 'Yes'
            pass

    for ((w, w2), w3), w4 in zip(zip(zip(text_tok[:-3], text_tok[1:-2]), text_tok[2:-1]), text_tok[3:]):
        if w in word_features and w2 in word_features and w3 in word_features and w4 in word_features:
            features['contains({}&&&{}&&&{}&&&{})'.format(w, w2, w3, w4)] = 'Yes'
            pass

    # features['contains(@@stars{})'.format(row.stars)] = 'Yes'
    features['@@@stars'] = row.stars
    features['@@@extreme_stars'] = False if 2 <= row.stars <= 4 else True
    features['@@@bus_stars'] = row['business_id']['stars']
    # features['@@@review_count']= "A lot" if row['business']['review_count']  else "A few"
    l = row['words']
    features['@@@review_length'] = "short" if l < 50 else "middle" if l < 150 else "long"
    features['@@@review_length50'] = "short" if l < 50 else "middle"
    features['@@@review_length100'] = "short" if l < 100 else "middle"
    features['@@@review_length150'] = "short" if l < 150 else "middle"
    features['@@@review_length35'] = "short" if l < 35 else "middle"
    features['@@@review_length75'] = "short" if l < 75 else "middle"

    rate = row['incorrect_words'] / row['words']

    features['@@@error_rate0.02'] = "good" if rate < 0.02 else "bad"
    features['@@@error_rate0.05'] = "good" if rate < 0.05 else "bad"
    features['@@@error_rate0.1'] = "good" if rate < 0.1 else "bad"
    features['@@@error_rate0.15'] = "good" if rate < 0.15 else "bad"
    features['@@@error_rate0.2'] = "good" if rate < 0.2 else "bad"

    features['@@@error_total5'] = "good" if rate < 5 else "bad"
    features['@@@error_total10<'] = "good" if rate < 10 else "bad"
    features['@@@error_total15'] = "good" if rate < 15 else "bad"
    features['@@@error_total20'] = "good" if rate < 20 else "bad"

    # not 100% haha
    # features['aaa'] = 'a' if row.useful > 0 else 'b'
    cos_sims = index[doc2vec(text)]
    for i, x in enumerate(cos_sims):
        features['@@@cos_sim4_{}'.format(i)] = 1 if x > 0.4 else 0
        features['@@@cos_sim6_{}'.format(i)] = 1 if x > 0.6 else 0
        features['@@@cos_sim8_{}'.format(i)] = 1 if x > 0.8 else 0
        features['@@@cos_sim9_{}'.format(i)] = 1 if x > 0.9 else 0
        features['@@@cos_sim95_{}'.format(i)] = 1 if x > 0.95 else 0

    return features


# In[36]:


# generate tuples: (features_dict, sentiment)
feature_sets = [(features(row), row.classification) for index, row in reviews.iterrows()]

# In[37]:


feature_sets[0]

# # Model training

# In[38]:


import random

random.shuffle(feature_sets)
half = int(len(feature_sets) / 2)
train_set, test_set = feature_sets[:half], feature_sets[half:]
half

# In[39]:


classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
print(nltk.classify.accuracy(classifier,
                             train_set))  # pridani jednotlivych slov tady snizi presnost jen na 65, je to ocekavane?

# In[40]:


classifier.show_most_informative_features(30)

# In[41]:


# classifier = nltk.DecisionTreeClassifier.train(train_set)
# print(nltk.classify.accuracy(classifier, test_set))
# print(nltk.classify.accuracy(classifier, train_set))


# # get feature matrix

# In[42]:


X, Y = [x[0] for x in feature_sets], [x[1] for x in feature_sets]

# In[43]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer

# In[44]:


X[0]

# In[45]:


cv_gain = CountVectorizer(max_df=0.95, min_df=2,
                          max_features=10000)  # WTF

# In[46]:


all_keys = [set(x.keys()) for x in X]

# In[47]:


import functools

all_fs = functools.reduce(lambda a, b: a.union(b), all_keys)
all_fs = list(all_fs)

# In[48]:


len(all_fs)


# In[49]:


def get_int(val):
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return val
    vals = {"Yes": 1, "No": 0, "middle": 1, "long": 2, "short": 0, "good": 1, "bad": 0}
    return vals[val]


# In[50]:


# X_matrix=[]
#
# for x in X:
#    row=[]
#    for key in all_fs:
#        if key in x:
#            row.append(get_int(x[key]))
#        else:
#            row.append(0)
#    X_matrix.append(row)


# In[51]:


import scipy

# In[52]:


row = []
x = X[0]

for key in all_fs:
    if key in x:
        row.append(get_int(x[key]))
    else:
        row.append(0)

X_matrix = scipy.sparse.lil_matrix([row])

i = 0
for x in X[1:]:
    row = []
    for key in all_fs:
        if key in x:
            row.append(get_int(x[key]))
        else:
            row.append(0)
    X_matrix = scipy.sparse.vstack((X_matrix, scipy.sparse.lil_matrix([row])), format='lil')
    i += 1
    # if i==1000:
    # break

# In[53]:


len(X)

# In[54]:


X_matrix

# ## logistic regression

# In[55]:


from sklearn.linear_model import LogisticRegression

# In[56]:


lr = LogisticRegression()

# In[57]:


half = int(len(X) / 2)
print(half)

# In[58]:


train_set_X, test_set_X = X_matrix[:half, :], X_matrix[half:, :]
train_set_Y, test_set_Y = Y[:half], Y[half:]

# In[59]:


lr.fit(train_set_X, train_set_Y)

# In[60]:


lr.score(test_set_X, test_set_Y)

# ## Dimension reduction - LSA - SVD

# In[55]:


from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import scale

# In[56]:


svd = TruncatedSVD(n_components=100)
# scale(X_matrix.tocsc())
svdMatrix = svd.fit_transform(X_matrix)

# In[57]:


feature_set_reduced = [(dict(enumerate(x)), y) for (x, y) in zip(svdMatrix, Y)]

# In[58]:


random.shuffle(feature_set_reduced)
half = int(len(feature_sets) / 2)
train_set, test_set = feature_sets[:half], feature_sets[half:]
half

# # training

# In[59]:


classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

# # get feature matrix

# In[60]:


X, Y = [x[0] for x in test_set], [x[1] for x in test_set]

# In[61]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer

# In[62]:


X[0]

# In[63]:


cv_gain = CountVectorizer(max_df=0.95, min_df=2,
                          max_features=10000)

# In[64]:


all_keys = [set(x.keys()) for x in X]

# In[65]:


import functools

all_fs = functools.reduce(lambda a, b: a.union(b), all_keys)
all_fs = list(all_fs)

# In[66]:


len(all_fs)


# In[67]:


def get_int(val):
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return val
    vals = {"Yes": 1, "No": 0, "middle": 1, "long": 2, "short": 0, "good": 1, "bad": 0}
    return vals[val]


# In[68]:


# X_matrix=[]
#
# for x in X:
#    row=[]
#    for key in all_fs:
#        if key in x:
#            row.append(get_int(x[key]))
#        else:
#            row.append(0)
#    X_matrix.append(row)


# In[69]:


import scipy

# In[70]:


row = []
x = X[0]

for key in all_fs:
    if key in x:
        row.append(get_int(x[key]))
    else:
        row.append(0)

X_matrix = scipy.sparse.lil_matrix([row])

i = 0
for x in X[1:]:
    row = []
    for key in all_fs:
        if key in x:
            row.append(get_int(x[key]))
        else:
            row.append(0)
    X_matrix = scipy.sparse.vstack((X_matrix, scipy.sparse.lil_matrix([row])))
    i += 1
    # if i==1000:
    # break

# In[71]:


len(X)

# In[72]:


X_matrix

# # information gaion

# In[73]:


res_gain = list(zip(all_fs, mutual_info_classif(X_matrix, Y, discrete_features=True)))

# In[74]:


# res_gain


# In[75]:


[(x, y) for (x, y) in res_gain if y > 0.0005]

# In[76]:


[(x, y) for (x, y) in res_gain if y > 0.001]

# In[77]:


sorted([(x, y) for (x, y) in res_gain if x[:3] == "@@@"])
