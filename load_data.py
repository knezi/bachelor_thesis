#!/bin/env python3
# TODO COMMENT AUTHOR
from itertools import islice

from enum import Enum
from functools import reduce

from typing import Iterator, List

import typing

import datetime as dt
import json
import nltk
import os
import pandas as pd
import random
from gensim import corpora
from gensim.similarities import Similarity
from matplotlib import pyplot
from nltk import TweetTokenizer
from pandas import DataFrame

import exceptions

data = 'just_restaurants.json'


class Plot:
    def __init__(self, path):
        self.path = path
        self.fig = pyplot.figure()

    def plot(self, x_data, y_data, name, x_title="", y_title="", title=""):
        # if title != "":
        # fig.suptitle(title)
        # pyplot.figure()
        # pyplot.plot(data)
        # pyplot.savefig(os.path.join(self.path, "{}.png".format(name)))

        self.fig.clf()
        ax = self.fig.subplots()
        ax.plot(x_data, y_data)
        self.fig.savefig(os.path.join(self.path, "{}.png".format(name)))


class SampleTypeEnum(Enum):
    TRAIN = 0
    TEST = 1
    CROSSVALIDATION = 2


class Sample:
    # train, test, crossvalidation
    # TODO COMM

    # format (features, data_line_from_pandas_data)
    def __init__(self) -> None:
        self._samples = dict()
        for x in SampleTypeEnum:
            self._samples[x] = []
        self._train_size = None

    def set_train(self,
                  sample: typing.List[typing.Tuple[dict, dict]]) \
            -> None:
        self._samples[SampleTypeEnum.TRAIN] = sample

    def set_test(self,
                 sample: typing.List[typing.Tuple[dict, dict]]) \
            -> None:
        self._samples[SampleTypeEnum.TEST] = sample

    def set_crossvalidation(self,
                            sample: typing.List[typing.Tuple[dict, dict]]) \
            -> None:
        self._samples[SampleTypeEnum.CROSSVALIDATION] = sample

    # training set
    def get_train_basic(self) \
            -> typing.List[typing.Tuple]:
        return self.get_data(SampleTypeEnum.TRAIN, 'classification')

    def get_train_extended(self, *args: str) \
            -> typing.List[typing.Tuple]:
        return self.get_data(SampleTypeEnum.TRAIN, *args)

    # testing set
    def get_test_basic(self) \
            -> typing.List[typing.Tuple]:
        return self.get_data(SampleTypeEnum.TEST, 'classification')

    def get_test_extended(self, *args: str) \
            -> object:
        return self.get_data(SampleTypeEnum.TEST, *args)

    # crossvalidating set
    def get_crossvalidate_basic(self) \
            -> typing.List[typing.Tuple]:
        return self.get_data(SampleTypeEnum.TEST, 'classification')

    def get_crossvalidate_extended(self, *args: str) \
            -> typing.List[typing.Tuple]:
        return self.get_data(SampleTypeEnum.TEST, *args)

    def get_data(self, set_no: int, *args: str)\
            -> typing.List[typing.Tuple]:
        res = [(row[0], *Sample._filter_row(row[1], args)) for row
                in self._samples[set_no]]
        if set_no == SampleTypeEnum.TRAIN and self._train_size is not None:
            res = res[0:self._train_size]
        return res

    def limit_train_size(self, size: int) -> None:
        if size > len(self._samples[SampleTypeEnum.TRAIN]):
            raise IndexError('Train set is not big enough.')
        self._train_size = size

    @staticmethod
    def _filter_row(row: pd.Series, args: typing.Tuple[str]) -> list:
        return [row[arg] for arg in args]


class Data:
    tokenizer: TweetTokenizer = nltk.tokenize.TweetTokenizer()

    def __init__(self, path_to_data: str, path_to_geneea_data: str):
        self._sample: Sample = Sample()

        # prepare statistics
        timestamp = dt.datetime.now().isoformat()
        self.statPath = os.path.join('graphs', timestamp)
        os.mkdir(self.statPath)
        self.stats = open(os.path.join(self.statPath, 'statistics'), 'w')

        # TODO
        # self.plot = Plot(self.statPath)
        # self.plot.plot([1,2], [1,2], 'a')
        # self.plot.plot([1,2], [4,2], 'b')

        # reading data in Pandas array - review per line
        self.path: str = path_to_data

        with open(self.path, 'r') as data, open(path_to_geneea_data, 'r') \
                as geneea:
            lines = []
            for d, g in zip(data, geneea):
                dj = json.loads(d)
                gj = json.loads(g)

                if dj['review_id'] != gj['id']:
                    raise exceptions.DataMismatchException(
                        'ids {} and {} do not match.'
                        .format(dj['review_id'], gj['id']))

                for key in gj:
                    if key == 'id':
                        continue
                    dj[key] = gj[key]

                lines.append(pd.DataFrame([dj]))

            panda_lines = pd.concat(lines).reset_index()

        # flattening
        panda_lines['business_review_count'] = \
            panda_lines['business_id'].map(lambda x: x['review_count'])
        panda_lines['attributes_count'] = \
            panda_lines['business_id'].map(lambda x: len(x['attributes']))

        # choosing only trustworthy restaurants
        self.data = panda_lines[(panda_lines['business_review_count'] > 50) &
                                (panda_lines['attributes_count'] > 10)].copy()

        self._prepare_tokens()

    def _tokenize(self, text: str) -> typing.List[str]:
        return self.tokenizer.tokenize(text.lower())

    def generate_sample(self, like_type: str, sample_size: int=None) \
            -> None:
        self._sample: Sample = Sample()

        sample: DataFrame = self._get_sample(like_type)
        sample_for_index: typing.List[str] = random.sample(list(
            sample[sample['classification'] == like_type]['text']
        ), 10)
        index: Similarity = self._generate_cosine_similarity_index(like_type,
                                                       sample_for_index)
        sample = [(self.features(row, index), row)
                  for _, row in sample.iterrows()]

        random.shuffle(sample)
        # TODO sample size
        train_size: int = int(len(sample) * 0.7)
        self._sample.set_train(sample[:train_size])
        self._sample.set_test(sample[train_size:])

        t=self._sample.get_train_extended('text')[0][1]
        tt=self._sample.get_test_extended('text')[0][1]

    def get_feature_matrix(self, like_type):
        pass

        # todo extract 1st, 3rd

    def get_feature_dict(self, like_type):
        pass

    def dump_fasttext_format(self, like_type: str, path_prefix: str) -> None:
        # print
        #  __label__classification
        #  features in the format _feature_value
        #  text
        train_p: str = '{}_train'.format(path_prefix)
        test_data_p: str = '{}_test_data'.format(path_prefix)
        test_lables_p: str = '{}_test_lables'.format(path_prefix)

        with open(train_p, 'w') as train, \
             open(test_data_p, 'w') as test_data, \
             open(test_lables_p, 'w') as test_lables:

            erase_nl_trans = str.maketrans({'\n': None})

            # train set
            train_sample: List[tuple] \
                = self._sample.get_train_extended('text',
                                                 'classification')
            for (fs, txt, clsf) in train_sample:
                print("__label__{} {} {}".format(clsf,
                                                  Data._convert_fs2fasttext(fs),
                                                  txt.translate(erase_nl_trans)
                                                 ), file=train)

            # test set
            test_sample: List[tuple]\
                = self._sample.get_test_extended('text',
                                                 'classification')
            for (fs, txt, clsf) in test_sample:
                print('{} {}'.format(Data._convert_fs2fasttext(fs),
                                     txt.translate(erase_nl_trans)
                                     ), file=test_data)
                print('__label__{}'.format(clsf), file=test_lables)

    @staticmethod
    def _convert_fs2fasttext(fs: dict) -> str:
        # convert dict of features
        # to iterable of strings in the format _feature_value
        feature_strings: Iterator[str]\
            = map(lambda k: '{}_{}'.format(k, fs[k]), fs)
        all_features_string: str = reduce(lambda s1, s2: '{} _{}'.format(s1, s2),
                                          feature_strings,
                                          '').strip()
        return all_features_string


    # TODO get dump to be able to observe data!

    def _get_sample(self, like_type: str) -> pd.DataFrame:
        pos = self.data[self.data[like_type] > 2].sample(frac=1).copy()
        pos['classification'] = like_type
        neg = self.data[self.data[like_type] == 0].sample(frac=1).copy()
        neg['classification'] = 'not-' + like_type
        sample = pd.concat([pos, neg])

        # chooses only a subset of features
        # TODO UPDATE? ?? WTF WHY IS THIS
        sample = sample[['text', like_type, 'classification', 'stars',
                         'business_id', 'words', 'incorrect_words',
                         'sentiment']] \
            .reset_index(drop=True)

        return sample

    def _prepare_tokens(self) -> None:
        texts_tokenized = (self._tokenize(row.text) for index, row
                           in self.data.iterrows())
        words_freqs = nltk.FreqDist(w.lower() for tokens in texts_tokenized
                                    for w in tokens)

        # TODO
        # for x in all_words:
        # print(all_words[x])

        # self.print('total number of words:', sum(all_words.values()))
        # self.print('unique words:', len(all_words))
        # self.print('words present only once:',
        # sum(c for c in all_words.values() if c == 1))
        # all_words.plot(30)

        # only the right frequencies
        self.words = words_freqs.copy()
        for w, count in words_freqs.items():
            if count > 200 or count == 20:
                # TODO Measure
                del self.words[w]

        self.words = frozenset(self.words.keys())

        # building a dictionary for counting cosine similarity
        texts = [[w for w in self._tokenize(row.text)
                  if w in self.words]
                 for _, row in self.data.iterrows()]
        self.gensim_dictionary = corpora.Dictionary(texts)

    def _generate_cosine_similarity_index(self, like_type, rand_samp):
        corpus = [self.gensim_dictionary.doc2bow(self._tokenize(t))
                  for t in rand_samp]

        index = Similarity(None, corpus, num_features=len(self.gensim_dictionary))
        return index

    def print(self, *line) -> None:
        print(*line, file=self.stats)

    def features(self, row, index):
        # todo predelat
        text = row.text
        txt_words = self._tokenize(text)
        features = {}

        for w in txt_words:
            if w in self.words:
                features['contains({})'.format(w)] = 'Yes'  # beze slov je to lepsi
                pass

        for w, w2 in zip(txt_words[:-1], txt_words[1:]):
            if w in self.words and w2 in self.words:
                features['contains({}&&&{})'.format(w, w2)] = 'Yes'
                pass

        for (w, w2), w3 in zip(zip(txt_words[:-2], txt_words[1:-1]), txt_words[2:]):
            if w in self.words and w2 in self.words and w3 in self.words:
                features['contains({}&&&{}&&&{})'.format(w, w2, w3)] = 'Yes'
                pass

        for ((w, w2), w3), w4 in zip(zip(zip(txt_words[:-3], txt_words[1:-2]), txt_words[2:-1]), txt_words[3:]):
            if w in self.words and w2 in self.words and w3 in self.words and w4 in self.words:
                features['contains({}&&&{}&&&{}&&&{})'.format(w, w2, w3, w4)] = 'Yes'
                pass

        # features['contains(@@stars{})'.format(row.stars)] = 'Yes'
        features['@@@stars'] = row.stars
        features['@@@extreme_stars'] = False if 2 <= row.stars <= 4 else True
        features['@@@bus_stars'] = row['business_id']['stars']
        # features['@@@review_count']= 'A lot' if row['business']['review_count']  else 'A few'
        l = row['words']
        features['@@@review_length'] = 'short' if l < 50 else 'middle' if l < 150 else 'long'
        features['@@@review_length50'] = 'short' if l < 50 else 'middle'
        features['@@@review_length100'] = 'short' if l < 100 else 'middle'
        features['@@@review_length150'] = 'short' if l < 150 else 'middle'
        features['@@@review_length35'] = 'short' if l < 35 else 'middle'
        features['@@@review_length75'] = 'short' if l < 75 else 'middle'

        rate = row['incorrect_words'] / row['words']

        features['@@@error_rate0.02'] = 'good' if rate < 0.02 else 'bad'
        features['@@@error_rate0.05'] = 'good' if rate < 0.05 else 'bad'
        features['@@@error_rate0.1'] = 'good' if rate < 0.1 else 'bad'
        features['@@@error_rate0.15'] = 'good' if rate < 0.15 else 'bad'
        features['@@@error_rate0.2'] = 'good' if rate < 0.2 else 'bad'

        features['@@@error_total5'] = 'good' if rate < 5 else 'bad'
        features['@@@error_total10<'] = 'good' if rate < 10 else 'bad'
        features['@@@error_total15'] = 'good' if rate < 15 else 'bad'
        features['@@@error_total20'] = 'good' if rate < 20 else 'bad'

        # not 100% haha WTF?
        # features['aaa'] = 'a' if row.useful > 0 else 'b'
        cos_sims = index[self.gensim_dictionary.doc2bow(self._tokenize(text))]
        for i, x in enumerate(cos_sims):
            features['@@@cos_sim4_{}'.format(i)] = 1 if x > 0.4 else 0
            features['@@@cos_sim6_{}'.format(i)] = 1 if x > 0.6 else 0
            features['@@@cos_sim8_{}'.format(i)] = 1 if x > 0.8 else 0
            # features['@@@cos_sim9_{}'.format(i)] = 1 if x > 0.9 else 0
            # features['@@@cos_sim95_{}'.format(i)] = 1 if x > 0.95 else 0

        return features


data = Data('data/data_sample.json', 'data/geneea_data_extracted_sample.json')
# data = Data('data/data.json', 'data/geneea_data_extracted.json')

data.generate_sample('useful')
data.dump_fasttext_format('useful', 'data/data_fasttext')
# fs = data.get_feature_dict('useful')
# print(1)

# like_type = 'useful'
# # like_type='funny'
# # like_type='cool'

# reviews = get_reviews(like_type)

# # In[25]:


# # In[28]:


# word_features = frozenset(words.keys())
# i = 0
# words_numbered = dict()
# for w in word_features:
# words_numbered[w] = i
# i += 1

# # In[29]:


# len(word_features)


# # In[30]:




# # In[35]:




# # In[36]:


# # generate tuples: (features_dict, sentiment)
# feature_sets = [(features(row), row.classification) for index, row in reviews.iterrows()]

# # In[37]:


# feature_sets[0]


# ########################### konec??
# # # Model training

# # In[38]:


# import random

# random.shuffle(feature_sets)
# half = int(len(feature_sets) / 2)
# train_set, test_set = feature_sets[:half], feature_sets[half:]
# half

# # In[39]:


# classifier = nltk.NaiveBayesClassifier.train(train_set)
# print(nltk.classify.accuracy(classifier, test_set))
# print(nltk.classify.accuracy(classifier,
# train_set))  # pridani jednotlivych slov tady snizi presnost jen na 65, je to ocekavane?

# # In[40]:


# classifier.show_most_informative_features(30)

# # In[41]:


# # classifier = nltk.DecisionTreeClassifier.train(train_set)
# # print(nltk.classify.accuracy(classifier, test_set))
# # print(nltk.classify.accuracy(classifier, train_set))


# # # get feature matrix

# # In[42]:


# X, Y = [x[0] for x in feature_sets], [x[1] for x in feature_sets]

# # In[43]:


# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_selection import mutual_info_classif
# from sklearn.feature_extraction.text import CountVectorizer

# # In[44]:


# X[0]

# # In[45]:


# cv_gain = CountVectorizer(max_df=0.95, min_df=2,
# max_features=10000)  # WTF

# # In[46]:


# all_keys = [set(x.keys()) for x in X]

# # In[47]:


# import functools

# all_fs = functools.reduce(lambda a, b: a.union(b), all_keys)
# all_fs = list(all_fs)

# # In[48]:


# len(all_fs)


# # In[49]:


# def get_int(val):
# if isinstance(val, int):
# return val
# if isinstance(val, float):
# return val
# vals = {'Yes': 1, 'No': 0, 'middle': 1, 'long': 2, 'short': 0, 'good': 1, 'bad': 0}
# return vals[val]


# # In[50]:


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


# # In[51]:


# import scipy

# # In[52]:


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
# X_matrix = scipy.sparse.vstack((X_matrix, scipy.sparse.lil_matrix([row])), format='lil')
# i += 1
# # if i==1000:
# # break

# # In[53]:


# len(X)

# # In[54]:


# X_matrix

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
