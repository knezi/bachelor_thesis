#!/bin/env python3
# TODO COMMENT AUTHOR
import sys
from collections import defaultdict

import yaml
from functools import reduce
from math import log2, ceil

from nltk.metrics import scores

from subprocess import CompletedProcess
from subprocess import PIPE

import nltk
import subprocess as sp
from typing import DefaultDict, Dict, List, Tuple

import classifiers
from classifiers.classifierbase import ClassifierBase
from load_data import Data, SampleTypeEnum, FeatureSetEnum, LikeTypeEnum
from statistics import DataGraph

## TODO
config_file = 'experiment.yaml'


def run_fasttext(prefix):
    """Capture output of run_fasttext.sh

    :param prefix: path that is passed to the script as an argument
    :return: dictionary of results property_name -> value
    """
    finished_process: CompletedProcess = sp.run(['./run_fasttext.sh', prefix],
                                                encoding='utf-8',
                                                stdout=PIPE)

    if finished_process.returncode != 0:
        print(f'fasttext with prefix {prefix} failed.', file=sys.stderr)

    return dict(map(lambda x: (x[0], float(x[1])),
                    map(lambda a: a.split(' '),
                        finished_process.stdout.strip().split('\n'))))


def compute_evaluation_scores(classifier: ClassifierBase,
                              data_set: List[Tuple[Dict, str]],
                              evaluated_class: LikeTypeEnum) \
        -> Dict[str, float]:
    """Evaluate classifier on dataset with common metrics.

    Namely calculates:
    precision, recall, accuracy, f-measure.

    And adds:
    tp, fp, np, tn (true/false positives/negatives)."""
    clas_scores: dict = {}
    correctly_classified: int = 0

    # metrics
    refsets: DefaultDict[str, set] = defaultdict(set)
    testsets: DefaultDict[str, set] = defaultdict(set)
    for i, (fs, label) in enumerate(data_set):
        refsets[label].add(i)
        classified = classifier.classify(fs)
        testsets[classified].add(i)

        if label == classified:
            correctly_classified += 1

    # we don't know how many and what are the values of negative classes
    # therefore we compute union of all and subtract positive elements
    negative_test: set = reduce(lambda a,b: a.union(b), testsets.values())\
                         - testsets[evaluated_class.value]
    negative_ref: set = reduce(lambda a,b: a.union(b), refsets.values())\
                        - refsets[evaluated_class.value]
    positive_test: set = testsets[evaluated_class.value]
    positive_ref: set = refsets[evaluated_class.value]

    clas_scores['tp'] = len(positive_test & positive_ref) / len(data_set)
    clas_scores['fp'] = len(positive_test & negative_ref) / len(data_set)
    clas_scores['tn'] = len(negative_test & negative_ref) / len(data_set)
    clas_scores['fn'] = len(negative_test & positive_ref) / len(data_set)

    clas_scores['precision'] = scores.precision(positive_ref,
                                                positive_test)
    clas_scores['recall'] = scores.recall(positive_ref,
                                          positive_test)
    clas_scores['f_measure'] = scores.f_measure(positive_ref,
                                                positive_test)
    # accuracy is true positives and true negatives over all instances
    clas_scores['accuracy'] =  correctly_classified / len(data_set)

    return clas_scores


with open(config_file, 'r') as cfg:
    experiments: dict = yaml.load(cfg)


train_size = data.generate_sample(LikeTypeEnum.USEFUL)

stats = DataGraph('summary', 'number of instances', 'percentage')

# texts_tokenized = (self._tokenize(row.text) for index, row
#                    in self.data.iterrows())
# words_freqs = nltk.FreqDist(w.lower() for tokens in texts_tokenized
#                             for w in tokens)
#
# # TODO statistics
# # for x in all_words:
# # print(all_words[x])
#
# # self.print('total number of words:', sum(all_words.values()))
# # self.print('unique words:', len(all_words))
# # self.print('words present only once:',
# # sum(c for c in all_words.values() if c == 1))
# # all_words.plot(30)
#
# # only the right frequencies
# self.gram_words = words_freqs.copy()
# for w, count in words_freqs.items():
#     if count > 200 or count == 20:
#         # TODO Measure
#         del self.gram_words[w]
#
# self.gram_words = frozenset(self.gram_words.keys())

# TODO this cannot exceed, but doesn't use up all data
for train_size in map(lambda x: 2**x, range(1, ceil(log2(train_size)))):
    data.limit_train_size(train_size)


    print(f'SIZE {train_size}')

    point: dict = dict()

    for ex in experiments['experiments']:
        train_set = data.get_feature_dict(SampleTypeEnum.TRAIN, ex['features'])
        test_set = data.get_feature_dict(SampleTypeEnum.TEST, ex['features'])
        cls: ClassifierBase \
            = getattr(classifiers, ex['classificator']).Classifier({})
        cls.train(train_set)

        evaluation: dict \
            = compute_evaluation_scores(cls, test_set, LikeTypeEnum.USEFUL)

        stats.add_points(train_size, evaluation)

data.plot(stats)


# FASTTEXT
# data.dump_fasttext_format('data/data_fasttext')
# out = run_fasttext('data/data_fasttext')
#
# point['fasttext accuracy'] = out['accuracy']
# point['fasttext precision'] = out['precision']
# point['fasttext recall'] = out['recall']
# f_mes = 2 * out['precision'] * out['recall'] / (out['precision'] + out['recall'])
# point['fasttext f_measure'] = f_mes



# pridani jednotlivych slov tady snizi presnost jen na 65, je to ocekavane?

# classifier.show_most_informative_features(30)

# # classifier = nltk.DecisionTreeClassifier.train(train_set)
# # print(nltk.classify.accuracy(classifier, test_set))
# # print(nltk.classify.accuracy(classifier, train_set))


# # ## logistic regression

# # In[55]:


# from sklearn.linear_model import LogisticRegression

# # In[56]:


# lr = LogisticRegression()

# # In[57]:


# half = int(len(X) / 2)
# print(half)

# # In[58]:


# train_set_X, test_set_X = X_matrix[:half, :], X_matrix[half:, :]
# train_set_Y, test_set_Y = Y[:half], Y[half:]

# # In[59]:


# lr.fit(train_set_X, train_set_Y)

# # In[60]:


# lr.score(test_set_X, test_set_Y)

# # ## Dimension reduction - LSA - SVD

# # In[55]:


# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import scale

# # In[56]:


# svd = TruncatedSVD(n_components=100)
# # scale(X_matrix.tocsc())
# svdMatrix = svd.fit_transform(X_matrix)

# # In[57]:


# feature_set_reduced = [(dict(enumerate(x)), y) for (x, y) in zip(svdMatrix, Y)]

# # In[58]:


# random.shuffle(feature_set_reduced)
# half = int(len(feature_sets) / 2)
# train_set, test_set = feature_sets[:half], feature_sets[half:]
# half

# # # training

# # In[59]:


# classifier = nltk.NaiveBayesClassifier.train(train_set)
# print(nltk.classify.accuracy(classifier, test_set))

# # # get feature matrix

# # In[60]:


# X, Y = [x[0] for x in test_set], [x[1] for x in test_set]

# # In[61]:


# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_selection import mutual_info_classif
# from sklearn.feature_extraction.text import CountVectorizer

# # In[62]:


# X[0]

# # In[63]:


# cv_gain = CountVectorizer(max_df=0.95, min_df=2,
# max_features=10000)

# # In[64]:


# all_keys = [set(x.keys()) for x in X]

# # In[65]:


# import functools

# all_fs = functools.reduce(lambda a, b: a.union(b), all_keys)
# all_fs = list(all_fs)

# # In[66]:


# len(all_fs)


# # In[67]:


# def get_int(val):
# if isinstance(val, int):
# return val
# if isinstance(val, float):
# return val
# vals = {'Yes': 1, 'No': 0, 'middle': 1, 'long': 2, 'short': 0, 'good': 1, 'bad': 0}
# return vals[val]


# # In[68]:


# # X_matrix=[]
# #
# # for x in X:
# #    row=[]
# #    for key in all_fs:
# #        if key in x:
# #            row.append(get_int(x[key]))
# #        else:
# #            row.append(0)
# #    X_matrix.append(row)


# # In[69]:


# import scipy

# # In[70]:


# row = []
# x = X[0]

# for key in all_fs:
# if key in x:
# row.append(get_int(x[key]))
# else:
# row.append(0)

# X_matrix = scipy.sparse.lil_matrix([row])

# i = 0
# for x in X[1:]:
# row = []
# for key in all_fs:
# if key in x:
# row.append(get_int(x[key]))
# else:
# row.append(0)
# X_matrix = scipy.sparse.vstack((X_matrix, scipy.sparse.lil_matrix([row])))
# i += 1
# # if i==1000:
# # break

# # In[71]:


# len(X)

# # In[72]:


# X_matrix

# # # information gaion

# # In[73]:


# res_gain = list(zip(all_fs, mutual_info_classif(X_matrix, Y, discrete_features=True)))

# # In[74]:


# # res_gain


# # In[75]:


# [(x, y) for (x, y) in res_gain if y > 0.0005]

# # In[76]:


# [(x, y) for (x, y) in res_gain if y > 0.001]

# # In[77]:


# sorted([(x, y) for (x, y) in res_gain if x[:3] == '@@@'])
