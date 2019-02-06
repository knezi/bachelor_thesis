#!/bin/env python3
# TODO COMMENT AUTHOR
from itertools import islice

from enum import Enum
from functools import reduce

from typing import Iterator, List, Union

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

    def plot(self, data, name, x_title="", y_title="", title=""):
        # if title != "":
        # fig.suptitle(title)
        # pyplot.figure()
        # pyplot.plot(data)
        # pyplot.savefig(os.path.join(self.path, "{}.png".format(name)))

        self.fig.clf()
        ax = self.fig.subplots()
        for d in data:
            ax.plot(*zip(*d))
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

    def set_data(self,
                 dataset: SampleTypeEnum,
                 sample: typing.List[typing.Tuple[dict, dict]]) \
            -> None:
        self._samples[dataset] = sample

    def get_data_basic(self, dataset: SampleTypeEnum) \
            -> typing.List[typing.Tuple]:
        return self.get_data(dataset, 'classification')

    def get_data(self, dataset: SampleTypeEnum, *args: str) \
            -> typing.List[typing.Tuple]:
        res = [(row[0], *Sample._filter_row(row[1], args)) for row
               in self._samples[dataset]]
        if dataset == SampleTypeEnum.TRAIN and self._train_size is not None:
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
    _plot: Plot
    statPath: str
    tokenizer: TweetTokenizer = nltk.tokenize.TweetTokenizer()

    def __init__(self, path_to_data: str, path_to_geneea_data: str):
        self._sample: Sample = Sample()

        # prepare statistics
        timestamp: str = dt.datetime.now().isoformat()
        self.statPath = os.path.join('graphs', timestamp)
        os.mkdir(self.statPath)
        self.stats = open(os.path.join(self.statPath, 'statistics'), 'w')

        # TODO
        self._plot = Plot(self.statPath)
        # self.plot.plot([1,2], [1,2], 'a')
        # self.plot.plot([1,2], [4,2], 'b')

        # reading data in Pandas array - review per line
        self.path: str = path_to_data

        with open(self.path, 'r') as data, open(path_to_geneea_data, 'r') \
                as geneea:
            lines: List[DataFrame] = []
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

    def generate_sample(self, like_type: str, sample_size: int = None) \
            -> int:
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
        self._sample.set_data(SampleTypeEnum.TRAIN, sample[:train_size])
        self._sample.set_data(SampleTypeEnum.TEST, sample[train_size:])

        return train_size

    def get_feature_matrix(self, like_type):
        pass

        # todo extract 1st, 3rd

    def get_feature_dict(self, dataset: SampleTypeEnum):
        return self._sample.get_data_basic(dataset)

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
                = self._sample.get_data(SampleTypeEnum.TRAIN,
                                        'text', 'classification')
            for (fs, txt, clsf) in train_sample:
                print("__label__{} {} {}".format(clsf,
                                                 Data._convert_fs2fasttext(fs),
                                                 txt.translate(erase_nl_trans)
                                                 ), file=train)

            # test set
            test_sample: List[tuple] \
                = self._sample.get_data(SampleTypeEnum.TEST,
                                        'text', 'classification')
            for (fs, txt, clsf) in test_sample:
                print('{} {}'.format(Data._convert_fs2fasttext(fs),
                                     txt.translate(erase_nl_trans)
                                     ), file=test_data)
                print('__label__{}'.format(clsf), file=test_lables)

    @staticmethod
    def _convert_fs2fasttext(fs: dict) -> str:
        # convert dict of features
        # to iterable of strings in the format _feature_value
        feature_strings: Iterator[str] \
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

    def limit_train_size(self, size: int) -> None:
        self._sample.limit_train_size(size)

    def plot(self, data, name, x_title="", y_title="", title=""):
    # just wrapper around plot
        self._plot.plot(data, name, x_title, y_title, title)
