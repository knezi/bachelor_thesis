#!/bin/env python3
"""Creates classes for loading and storing data.

SampleTypeEnum - distinguishing between TRAIN, TEST, CROSSVALIDATION
Sample - container holding separately train,test,cv data
Data - class for loading, generating inferred and storing (in Sample) data
"""
from enum import Enum, unique, auto
from functools import reduce

from typing import Iterator, List, Tuple, Dict, Set

import datetime as dt
import json
import nltk
import os
import pandas as pd
import random
from gensim import corpora
from gensim.similarities import Similarity
from nltk import TweetTokenizer
from pandas import DataFrame, Series

import exceptions
from statistics import PointsPlot, Statistics, DataGraph


@unique
class SampleTypeEnum(Enum):
    """Enum used for denoting the use of data (TRAIN, TEST, CROSSVALIDATION)"""
    TRAIN = 0
    TEST = 1
    CROSSVALIDATION = 2


class Sample:
    """Store data for different usage and allow access to them.

    __init__ - construct empty objects
    set_data - set data to one of SampleTypeEnum
    get_data_basic - return data of SampleTypeEnum with text&classification
    get_data - return data of SampleTypeEnum with text and specified columns
    limit_train_size - set only [:n] rows accessible"""

    def __init__(self) -> None:
        """Construct empty objects."""
        self._samples = dict()
        for x in SampleTypeEnum:
            self._samples[x] = []
        self._train_size = None

    def set_data(self,
                 dataset: SampleTypeEnum,
                 sample: List[Tuple[dict, Series]]) \
            -> None:
        """Set data to the given set

        Data must be list of instances. Each element being a tuple of
        feature dictionary and attribute dictionary

        :param dataset: given type - SampleTypeEnum
        :param sample:  the actual sample being set
        :return: None
        """
        self._samples[dataset] = sample

        # TODO REMOVE adding classified feature
        def add_f(row):
            row[0]['classification'] = row[1]['classification']
            return row
        # self._samples[dataset] = [add_f(x) for x in self._samples[dataset]]

    def get_data_basic(self, dataset: SampleTypeEnum) \
            -> List[Tuple]:
        """Only calls self.get_data(dataset, 'classification')

        :param dataset: wanted type - SampleTypeEnum
        :return: wanted dataset
        """
        return self.get_data(dataset, 'classification')

    def get_data(self, dataset: SampleTypeEnum, *args: str) \
            -> List[Tuple]:
        """Return list of instances represented by tuples.

        Each tuple contains feature dictionary of an instance at the index 0.
        The remaining indices are attribute values defined by
        the remaining arguments passed to this function.

        Feature dictionary is mapping feature(str) -> value(scalar type)

        :param dataset: wanted type
        :param args: columns being added to the end of each tuple
        :return: wanted dataset
        """
        res = [(row[0], *Sample._filter_row(row[1], args)) for row
               in self._samples[dataset]]
        if dataset == SampleTypeEnum.TRAIN and self._train_size is not None:
            res = res[0:self._train_size]
        return res

    def limit_train_size(self, size: int) -> None:
        """Limit the accessible part of train data to [:size]

        :param size: First `size` elements of train data will be used.
        """
        if size > len(self._samples[SampleTypeEnum.TRAIN]):
            raise IndexError('Train set is not big enough.')
        self._train_size = size

    @staticmethod
    def _filter_row(row: pd.Series, args: Tuple[str]) -> tuple:
        """Convert a given row to a tuple containing only the specified columns.

        :param row: panda Series of data
        :param args: tuple of wanted columns
        :return: resulting tuple
        """
        return tuple(row[arg] for arg in args)


@unique
class FeatureSet(Enum):
    """Enum used for defining sets of features"""
    UNIGRAMS     = auto()
    BIGRAMS      = auto()
    TRIGRAMS     = auto()
    FOURGRAMS    = auto()
    STARS        = auto()
    REVIEWLEN    = auto()
    SPELLCHECK   = auto()
    COSINESIM    = auto()


class Data:
    """Load data from specified files to memory and make it accessible.

    __init__ - take paths and load data to memory
    generate_sample - create a sample stored internally
TODO
    """
    _statistics: Statistics
    _statPath: str
    tokenizer: TweetTokenizer = nltk.tokenize.TweetTokenizer()

    def __init__(self, path_to_data: str, path_to_geneea_data: str):
        """Load data to memory and init class.

        :param path_to_data: JSON-line file as given from denormalise.sh
        :param path_to_geneea_data: extracted data as output of?? TODO
        """
        self._sample: Sample = Sample()

        # prepare statistics
        timestamp: str = dt.datetime.now().isoformat()
        self._statPath = os.path.join('graphs', timestamp)
        os.mkdir(self._statPath)
        self.stats = open(os.path.join(self._statPath, 'statistics'), 'w')

        self._statistics = Statistics(self._statPath)

        # reading data into Pandas array - review per line
        self.path: str = path_to_data

        # self.path contain text, desired classification and some other features
        # Instances correspond line-by-line with file path_to_geneea_data
        # which contain extra linguistics features extracted from text
        # this loop joins them together to one panda array
        with open(self.path, 'r') as data, open(path_to_geneea_data, 'r') \
                as geneea:
            lines: List[DataFrame] = []
            for d, g in zip(data, geneea):
                dj = json.loads(d)
                gj = json.loads(g)

                # check line-by-line correspondence
                if dj['review_id'] != gj['id']:
                    raise exceptions.DataMismatchException(
                        f'ids {dj["review_id"]} and {gj["id"]} do not match.')

                for key in gj:
                    if key == 'id':
                        continue
                    dj[key] = gj[key]

                lines.append(pd.DataFrame([dj]))

            panda_lines: pd.DataFrame = pd.concat(lines).reset_index()

        # flattening - all properties need to be only scalar values
        panda_lines['business_review_count'] = \
            panda_lines['business_id'].map(lambda x: x['review_count'])
        panda_lines['attributes_count'] = \
            panda_lines['business_id'].map(lambda x: len(x['attributes']))

        # choosing only trustworthy restaurants
        self.data = panda_lines[(panda_lines['business_review_count'] > 50) &
                                (panda_lines['attributes_count'] > 10)].copy()

        self._prepare_tokens()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize given string with nltk tokenizer.

        :param text: text to be tokenized
        :return: list of words
        """
        return self.tokenizer.tokenize(text.lower())

    def generate_sample(self, like_type: str, fs_selection: Set[FeatureSet]) \
            -> int:
        """Generate sample from all data available of the particular like type.

        Create train and test set and set them as the current sample used
        by methods returning instances of data (get_feature_matrix,
        get_feature_dict, dump_fasttext_format). It doesn't create
        crossvalidation set as of now.

        :param like_type: string name of the predicted attribute
                          possible values useful, funny, cool
                      It is returned by _generate_cosine_similarity_index.
        :param fs_selection: set specifying which features will be used.
                             each element is of type FeatureSet which
                             corresponds to a subset of features.
        :return: int the size of train set
        """
        self._sample: Sample = Sample()

        # get all usable data for the given like type
        raw_sample: DataFrame = self._get_raw_sample(like_type)

        # build index of positive instances for computing cosine similarity
        # features expressing cosine distance to all instances in the index
        # are later added to each instance
        sample_for_index: List[str] = random.sample(list(
            raw_sample[raw_sample['classification'] == like_type]['text']
        ), 10)
        index: Similarity = self._generate_cosine_similarity_index(sample_for_index)

        # computing features for instances and creating datastructures
        # for samples to be given further
        sample: List[Tuple[Dict, Series]]\
            = [(self.generate_features(row, index, fs_selection), row)
               for _, row in raw_sample.iterrows()]

        # splitting data into sample sets train and test (7:3 ratio)
        random.shuffle(sample)
        train_size: int = int(len(sample) * 0.7)
        self._sample.set_data(SampleTypeEnum.TRAIN, sample[:train_size])
        self._sample.set_data(SampleTypeEnum.TEST, sample[train_size:])

        return train_size

    def get_feature_matrix(self, like_type):
        pass

        # todo extract 1st, 3rd

    def get_feature_dict(self, dataset: SampleTypeEnum) -> List[tuple]:
        """TODO"""
        return self._sample.get_data_basic(dataset)

    def dump_fasttext_format(self, path_prefix: str) -> None:
        """Create training & testing files for fasttext from the current sample.

        Train set is written into file {path_prefix}_train instance per line
        in format:
            __label__classification \
            features in the format _feature_value \
            text

        Test set is divided into two files:
            {path_prefix}_test_data - format same as for the training file
                without __label__classification

            {path_prefix}_test_lables - file containing only lables
                __label__classification
                Order of lines corresponds to first test file

        :param path_prefix: prefix used for all three files"""
        train_p: str = f'{path_prefix}_train'
        test_data_p: str = f'{path_prefix}_test_data'
        test_lables_p: str = f'{path_prefix}_test_lables'

        with open(train_p, 'w') as train, \
                open(test_data_p, 'w') as test_data, \
                open(test_lables_p, 'w') as test_lables:

            # instance per line, newline is forbidden
            erase_nl_trans = str.maketrans({'\n': None})

            # train set
            train_sample: List[tuple] \
                = self._sample.get_data(SampleTypeEnum.TRAIN,
                                        'text', 'classification')
            for (fs, txt, clsf) in train_sample:
                # is this string better to turn into f-string too?
                print("__label__{} {} {}".format(clsf,
                                                 Data._convert_fs2fasttext(fs),
                                                 txt.translate(erase_nl_trans)
                                                 ), file=train)

            # test set
            test_sample: List[tuple] \
                = self._sample.get_data(SampleTypeEnum.TEST,
                                        'text', 'classification')
            for (fs, txt, clsf) in test_sample:
                print(Data._convert_fs2fasttext(fs) + ' ' +
                      txt.translate(erase_nl_trans), file=test_data)
                print(f'__label__{clsf}', file=test_lables)

    @staticmethod
    def _convert_fs2fasttext(fs: dict) -> str:
        # convert dict of features
        # to iterable of strings in the format _feature_value
        feature_strings: Iterator[str] \
            = map(lambda k: f'{k}_{fs[k]}', fs)
        all_features_string: str = reduce(lambda s1, s2: f'{s1} _{s2}',
                                          feature_strings,
                                          '').strip()
        return all_features_string

    # TODO get dump to be able to observe data!

    def _get_raw_sample(self, like_type: str) -> pd.DataFrame:
        """Return all usable raw data of the given like type in PandaSeries.

        All lines with at least two likes are classified as positive,
        all with zero negative. Lines with only one like are disregarded.

        :param like_type: the type of like being classified
        :return: panda series containing text, likes (number of likes),
         classification, other data
        """
        pos = self.data[self.data[like_type] > 2].sample(frac=1).copy()
        pos['classification'] = like_type
        neg = self.data[self.data[like_type] == 0].sample(frac=1).copy()
        neg['classification'] = 'not-' + like_type
        sample: pd.DataFrame = pd.concat([pos, neg])

        # chooses only a subset of features for memory reasons
        sample = sample[['text', like_type, 'classification', 'stars',
                         'business_id', 'words', 'incorrect_words',
                         'sentiment']] \
            .reset_index(drop=True)
        
        sample.rename(columns={like_type: 'likes'})

        return sample

    def _prepare_tokens(self) -> None:
        """Building lists of words for features and gensim dictionary."""
        texts_tokenized = (self._tokenize(row.text) for index, row
                           in self.data.iterrows())
        words_freqs = nltk.FreqDist(w.lower() for tokens in texts_tokenized
                                    for w in tokens)

        # TODO statistics
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

    def _generate_cosine_similarity_index(self, rand_samp: List[str])\
            -> Similarity:
        """Built index from the given rand_samp for computing cosine similarity.

        :param rand_samp: Index will be built out of these strings.
        :return: index
        """
        corpus = [self.gensim_dictionary.doc2bow(self._tokenize(t))
                  for t in rand_samp]

        index: Similarity = Similarity(None, corpus,
                                       num_features=len(self.gensim_dictionary))
        return index

    def print(self, *line) -> None:
        """Log into a statistics file.

        :param line: These arguments will be passed as are to print
        """
        print(*line, file=self.stats)

    def generate_features(self, row: pd.Series, index: Similarity,
                          fs_selection: Set[FeatureSet])\
            -> Dict[str, any]:
        """Create dictionary of features from the given row.

        :param row: Data of a review
        :param index: Gensim index for computing cosine similarity.
                      It is returned by _generate_cosine_similarity_index.
        :param fs_selection: set specifying which features will be used.
                             each element is of type FeatureSet which
                             corresponds to a subset of features.
        :return: Feature dict name_of_feature -> value
        """
        text = row.text
        txt_words = self._tokenize(text)
        features = {}

        # GENERAL NON-TEXTUAL FEATURES
        if FeatureSet.STARS in fs_selection:
            # TODO convert to float/int + add no?
            features[f'stars({row.stars})'] = 'Yes'
            features['stars'] = row.stars
            features['extreme_stars'] = False if 2 <= row.stars <= 4 else True
            features['bus_stars'] = row['business_id']['stars']

        # TEXTUAL FEATURES
        # N-GRAMS
        if FeatureSet.UNIGRAMS in fs_selection:
            for w in txt_words:
                if w in self.words:
                    features[f'contains({w})'] = 'Yes'

        if FeatureSet.BIGRAMS in fs_selection:
            for w, w2 in zip(txt_words, txt_words[1:]):
                if w in self.words and w2 in self.words:
                    features[f'contains({w}&{w2})'] = 'Yes'

        if FeatureSet.TRIGRAMS in fs_selection:
            for w, w2, w3 in zip(txt_words, txt_words[1:], txt_words[2:]):
                if w in self.words and w2 in self.words and w3 in self.words:
                    features[f'contains({w}&{w2}&{w3})'] = 'Yes'

        if FeatureSet.FOURGRAMS in fs_selection:
            for w, w2, w3, w4 in zip(txt_words, txt_words[1:], txt_words[2:], txt_words[3:]):
                if w in self.words and w2 in self.words and w3 in self.words and w4 in self.words:
                    features[f'contains({w}&{w2}&{w3}&{w4})'] = 'Yes'

        # MISC
        if FeatureSet.REVIEWLEN in fs_selection:
            # features['@@@review_count']= 'A lot' if row['business']['review_count'] TODO add constant else 'A few'
            l = row['words']
            features['review_length'] = 'short' if l < 50 else 'middle' if l < 150 else 'long'
            features['review_length50'] = 'short' if l < 50 else 'middle'
            features['review_length100'] = 'short' if l < 100 else 'middle'
            features['review_length150'] = 'short' if l < 150 else 'middle'
            features['review_length35'] = 'short' if l < 35 else 'middle'
            features['review_length75'] = 'short' if l < 75 else 'middle'

        if FeatureSet.SPELLCHECK in fs_selection:
            rate = row['incorrect_words'] / row['words']

            features['error_rate0.02'] = 'good' if rate < 0.02 else 'bad'
            features['error_rate0.05'] = 'good' if rate < 0.05 else 'bad'
            features['error_rate0.1'] = 'good' if rate < 0.1 else 'bad'
            features['error_rate0.15'] = 'good' if rate < 0.15 else 'bad'
            features['error_rate0.2'] = 'good' if rate < 0.2 else 'bad'

            features['error_total5'] = 'good' if rate < 5 else 'bad'
            features['error_total10<'] = 'good' if rate < 10 else 'bad'
            features['error_total15'] = 'good' if rate < 15 else 'bad'
            features['error_total20'] = 'good' if rate < 20 else 'bad'

        if FeatureSet.COSINESIM in fs_selection:
            cos_sims = index[self.gensim_dictionary.doc2bow(self._tokenize(text))]
            for i, x in enumerate(cos_sims):
                features[f'cos_sim0.4_{i}'] = True if x > 0.4 else False
                features[f'cos_sim0.6_{i}'] = True if x > 0.6 else False
                features[f'cos_sim0.8_{i}'] = True if x > 0.8 else False
                features[f'cos_sim0.9_{i}'] = True if x > 0.9 else False
                features[f'cos_sim0.95_{i}'] = True if x > 0.95 else False

        # TODO linguistics features
        # sentiment
        # entities

        return features

    def limit_train_size(self, size: int) -> None:
        """Directly call Sample.limit_train_size.

        :param size: size of train data
        """
        self._sample.limit_train_size(size)

    def plot(self, data: DataGraph) -> None:
        """Plot given DataGraph.

        :param data: instance of DataGraph to be plotted"""
        self._statistics.plot(data)
