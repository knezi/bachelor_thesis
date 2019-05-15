#!/bin/env python3
"""Creates classes for loading and storing data.

SampleTypeEnum - distinguishing between TRAIN, TEST, CROSSVALIDATION
Sample - container holding separately train,test,cv data
Data - class for loading, generating inferred and storing (in Sample) data
"""
from enum import Enum, unique, auto

from typing import List, Tuple, Dict, Set

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
from geneea.analyzer.model import f2converter
from statistics import DataLine, Statistics, DataGraph


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
        self._samples: Dict[SampleTypeEnum, List[Series]] = dict()
        for x in SampleTypeEnum:
            self._samples[x] = []
        self._train_size = None

    def set_data(self,
                 dataset: SampleTypeEnum,
                 sample: List[Series]) \
            -> None:
        """Set data to the given set

        Data must be list of instances. Each element being panda Series

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

    def get_data(self, dataset: SampleTypeEnum) \
            -> List[Series]:
        """Return list of instances represented by panda Series.

        :param dataset: wanted type
        :return: wanted dataset
        """
        size = self._train_size if (
                dataset == SampleTypeEnum.TRAIN and
                self._train_size is not None) \
            else len(self._samples[dataset])
        return self._samples[dataset][:size]

    def limit_train_size(self, size: int) -> None:
        """Limit the accessible part of train data to [:size]

        :param size: First `size` elements of train data will be used.
        """
        if size > len(self._samples[SampleTypeEnum.TRAIN]):
            raise IndexError('Train set is not big enough.')
        self._train_size = size


@unique
class FeatureSetEnum(Enum):
    """Enum used for defining sets of features"""
    UNIGRAMS = auto()
    BIGRAMS = auto()
    TRIGRAMS = auto()
    FOURGRAMS = auto()
    STARS = auto()
    REVIEWLEN = auto()
    SPELLCHECK = auto()
    COSINESIM = auto()


@unique
class LikeTypeEnum(Enum):
    """Enum used for denoting which class is classified"""
    USEFUL = 'useful'
    COOL = 'cool'
    FUNNY = 'funny'


class Incrementer:
    """Calling this function returns a number incremented by one per call.

    First returned number is 1, so the returned number is how many this
    instance has been already called."""

    def __init__(self):
        self.state: int = 0

    def __call__(self) -> int:
        self.state += 1
        return self.state


class Data:
    """Load data from specified files to memory and make it accessible.

    __init__ - take paths and load data to memory
    generate_sample - create a sample stored internally
TODO
    """
    # only words contained in this set will be used
    # when generating n-gram features
    used_gram_words: Set[str]
    # only these entities will be used when generating entity features
    used_entities: Set[str]
    _statistics: Statistics
    _statPath: str
    tokenizer: TweetTokenizer = nltk.tokenize.TweetTokenizer()

    def __init__(self, path_to_data: str, path_to_geneea_data: str):
        """Load data to memory and init class.

        :param path_to_data: JSON-line file as given from denormalise.sh
        :param path_to_geneea_data: extracted data as output of?? TODO
        """
        self._sample: Sample = Sample()
        self.index: Similarity = None

        # prepare statistics
        timestamp: str = dt.datetime.now().isoformat()
        self._statPath = os.path.join('graphs', timestamp)
        os.mkdir(self._statPath)
        self.stats = open(os.path.join(self._statPath, 'statistics'), 'w')

        self._statistics = Statistics(self._statPath)

        # reading data into Pandas array - review per line
        self.path: str = path_to_data

        # set variables controlling feature creation
        self.used_gram_words = None
        self.used_entities = None

        # self.path contain text, desired classification and some other features
        # Instances correspond line-by-line with file path_to_geneea_data
        # which contain extra linguistics features extracted from text
        # this loop joins them together to one panda array
        with open(self.path, 'r') as data, open(path_to_geneea_data, 'r') \
                as geneea:
            lines: List[DataFrame] = []
            for d, g in zip(data, geneea):
                dj = json.loads(d)
                gx3 = f2converter.fromDict(json.loads(g))

                # check line-by-line correspondence
                if dj['review_id'] != gx3.docId:
                    raise exceptions.DataMismatchException(
                        f'ids {dj["review_id"]} and {gx3.docId} do not match.')

                dj['sentiment'] = gx3.sentiment.label
                dj['entities'] = [ent.stdForm for ent in gx3.entities]

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

    def __del__(self) -> None:
        """Close open FileDescriptor for stat file."""
        self.stats.close()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize given string with nltk tokenizer.

        :param text: text to be tokenized
        :return: list of words
        """
        return self.tokenizer.tokenize(text.lower())

    def generate_sample(self, like_type: LikeTypeEnum) -> int:
        """Generate sample from all data available of the particular like type.

        Create train and test set and set them as the current sample used
        by methods returning instances of data. It doesn't create
        crossvalidation set as of now.

        :param like_type: class being classified
                      It is returned by _generate_cosine_similarity_index.
        :return: int the size of train set
        """
        self._sample: Sample = Sample()

        # get all usable data for the given like type
        raw_sample: DataFrame = self._get_raw_sample(like_type)

        # build index of positive instances for computing cosine similarity
        # features expressing cosine distance to all instances in the index
        # are later added to each instance
        sample_for_index: List[str] = random.sample(list(
            raw_sample[raw_sample['classification'] == like_type.value]['text']
        ), 10)
        self.index = self._generate_cosine_similarity_index(sample_for_index)

        # creating datastructures for samples to be given further
        sample: List[Series] = [row for _, row in raw_sample.iterrows()]

        # splitting data into sample sets train and test (7:3 ratio)
        random.shuffle(sample)
        train_size: int = int(len(sample) * 0.7)
        self._sample.set_data(SampleTypeEnum.TRAIN, sample[:train_size])
        self._sample.set_data(SampleTypeEnum.TEST, sample[train_size:])

        return train_size

    # @staticmethod
    # def _convert_feature_to_int(self, feature_value) -> int:
    #     """Return integer represeantation of various attribute values.
    #
    #     :param feature_value: integers are returned as are
    #                           string converted with feature_convert_table"""
    #     if isinstance(feature_value, int):
    #         return feature_value
    #     return Data.feature_convert_table[feature_value]

    # def get_feature_matrix(self, dataset: SampleTypeEnum) \
    #         -> Tuple[Tuple[str], List[List[int]]]:
    #     ### TODO tohle nefunguje presunout mimo viz pozn.md
    #     """Return feature matrix, columns are attributes, rows instances.
    #     Last column is classification class.
    #
    #     Attr values are converted with function _convert_feature_to_int.
    #     Matrix is represented as List [instance = List [ attr_value = Int] ]
    #     :param dataset: which dataset from sample is used
    #     :return: (header, matrix)
    #              header being tuple of string
    #              matrix in the format specified above"""
    #     # each instance is tuple ({feature dict}, 'classification')
    #     raw_data: List[tuple] = self._sample.get_data_basic(dataset)
    #
    #     all_keys: List[Set[str]] = [set(x[0].keys()) for x in raw_data]
    #     # get all feature names, missing values are filled with 0
    #     # convert to tuple to preserve the order
    #     all_fs: Tuple[str] = tuple(reduce(lambda a, b: a.union(b), all_keys))
    #
    #     def new_incremental_dict():
    #         """Return a dictionary where accessed non-existing value is len(dict)+1
    #         """
    #         inc: Incrementer = Incrementer()
    #         return defaultdict(inc)
    #
    #     # this creates a two-dimensional dictionary
    #     # It is for converted feature values into integers.
    #     # Every time a new value of a feature is accessed, it is given
    #     # a unique int identifier
    #     # each Feature is incremented separately
    #     # 0 is left for missing values
    #     # usage: feature_convert_table[f1][v1] gives an int representation
    #     # of feature f1 with value v1, it is always the same
    #     feature_convert_table: Dict[str, Dict] \
    #         = defaultdict(new_incremental_dict)
    #
    #     matrix: List[List[int]] = []
    #
    #     # iterating through instances
    #     for fs, cls in raw_data:
    #         row: List[int] = []
    #         # iterating through features in the specified order
    #         for key in all_fs:
    #             if key in fs:
    #                 row.append(feature_convert_table[key][fs[key]])
    #             else:
    #                 row.append(0)
    #
    #         # adding classification to the last column
    #         row.append(feature_convert_table['classification'][cls])
    #
    #         matrix.append(row)
    #
    #     # convert to sparse matrix, is this needed?? TODO
    #     # import scipy
    #     # row = []
    #     # x = X[0]
    #     # X_matrix = scipy.sparse.lil_matrix([row])
    #     # i = 0
    #     # for x in X[1:]:
    #     # row = []
    #     # for key in all_fs:
    #     # if key in x:
    #     # row.append(get_int(x[key]))
    #     # else:
    #     # row.append(0)
    #     # X_matrix = scipy.sparse.vstack((X_matrix, scipy.sparse.lil_matrix([row])), format='lil')
    #     # i += 1
    #     # # if i==1000:
    #     # # break
    #     # X_matrix
    #
    #     # header must also contain the classification column
    #     header: Tuple[str] = (*all_fs, 'classification')
    #     return header, matrix

    def get_feature_dict(self, dataset: SampleTypeEnum,
                         fs_selection: Set[FeatureSetEnum],
                         extra_columns: Tuple[str] = ()) -> List[tuple]:
        """Return list of instances, attributes being represented by dict.

        Each instance is a tuple of
        (feature dictionary {'feature' -> 'value'}, classification,
        columns specified in *extra_columns)

        :param dataset: data set being returned
        :param fs_selection: set specifying which features will be used.
            each element is of type FeatureSet which
            corresponds to a subset of features.
        :param extra_columns: any extra columns from raw data that will be
            appended to the end of each tuple
        :return: list of instances represented by tuples, each tuple being:
        (dict of features, classification: str, extra_columns)"""

        sample: List[Series] = self._sample.get_data(dataset)

        res: List[tuple] = [
            (self.generate_features(row, fs_selection),
             row['classification'],
             *self._filter_row(row, extra_columns))
            for row in sample]

        return res

    @staticmethod
    def _filter_row(row: pd.Series, args: Tuple[str]) -> tuple:
        """Convert a given row to a tuple containing only the specified columns.

        :param row: panda Series of data
        :param args: tuple of wanted columns
        :return: resulting tuple
        """
        return tuple(row[arg] for arg in args)

    # def get_raw_data(self, dataset: SampleTypeEnum, *attributes: str) \
    #         -> List[Tuple]:
    #     """Return raw data from specified dataset in a list of tuples.
    #
    #     This method is used only for the purpose of observing data.
    #
    #     Each instance is a tuple of attributes specified in the argument in
    #     that order.
    #
    #     :param dataset: returned dataset
    #     :param attributes: tuple of attributes as named in JSON
    #     :return: list of instances in the dataset
    #     """
    #     return list(map(lambda row: row[1:],
    #                     self._sample.get_data(dataset, *attributes)))
    #

    def _get_raw_sample(self, like_type: LikeTypeEnum) -> pd.DataFrame:
        """Return all usable raw data of the given like type in PandaSeries.

        All lines with at least two likes are classified as positive,
        all with zero negative. Lines with only one like are disregarded.

        :param like_type: class beying classified
        :return: panda series containing text, likes (number of likes),
         classification, other data
        """
        pos = self.data[self.data[like_type.value] > 2].sample(frac=1).copy()
        pos['classification'] = like_type.value
        neg = self.data[self.data[like_type.value] == 0].sample(frac=1).copy()
        neg['classification'] = 'not-' + like_type.value
        sample: pd.DataFrame = pd.concat([pos, neg])

        # chooses only a subset of features for memory reasons
        sample = sample[['text', like_type.value, 'classification', 'stars',
                         'business_id', 'words', 'incorrect_words',
                         'sentiment']] \
            .reset_index(drop=True)

        sample.rename(columns={like_type.value: 'likes'})

        return sample

    def _prepare_tokens(self) -> None:
        """Building lists of words for features and gensim dictionary."""
        # TODO REBUILD
        # MOVE dictionary building somewhere else? Outside this classs
        # for not needing to precompute gram_words?
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
        gram_words = words_freqs.copy()
        for w, count in words_freqs.items():
            if count > 200 or count == 20:
                # TODO Measure
                del gram_words[w]

        gram_words = frozenset(gram_words.keys())

        # building a dictionary for counting cosine similarity
        texts = [[w for w in self._tokenize(row.text)
                  if w in gram_words]
                 for _, row in self.data.iterrows()]
        self.gensim_dictionary = corpora.Dictionary(texts)

    def _generate_cosine_similarity_index(self, rand_samp: List[str]) \
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

    def generate_features(self, row: pd.Series,
                          fs_selection: Set[FeatureSetEnum]) \
            -> Dict[str, any]:
        """Create dictionary of features from the given row.

        :param row: Data of a review
        :param fs_selection: set specifying which features will be used.
                             each element is of type FeatureSet which
                             corresponds to a subset of features.
        :return: Feature dict name_of_feature -> value
        """
        text = row.text
        txt_words = self._tokenize(text)
        features = {}

        # GENERAL NON-TEXTUAL FEATURES
        if FeatureSetEnum.STARS in fs_selection:
            # TODO convert to float/int + add no?
            features[f'stars({row.stars})'] = 'Yes'
            features['stars'] = row.stars
            features['extreme_stars'] = False if 2 <= row.stars <= 4 else True
            features['bus_stars'] = row['business_id']['stars']

        # TEXTUAL FEATURES
        # N-GRAMS
        # TODO squeze this into a funciton call for all at once
        if FeatureSetEnum.UNIGRAMS in fs_selection:
            if self.used_gram_words is None:
                raise exceptions.InsufficientDataException('Word set not defined.')
            for w in txt_words:
                if w in self.used_gram_words:
                    features[f'contains({w})'] = 'Yes'

        if FeatureSetEnum.BIGRAMS in fs_selection:
            if self.used_gram_words is None:
                raise exceptions.InsufficientDataException('Word set not defined.')
            for w, w2 in zip(txt_words, txt_words[1:]):
                if w in self.used_gram_words and w2 in self.used_gram_words:
                    features[f'contains({w}&{w2})'] = 'Yes'

        if FeatureSetEnum.TRIGRAMS in fs_selection:
            if self.used_gram_words is None:
                raise exceptions.InsufficientDataException('Word set not defined.')
            for w, w2, w3 in zip(txt_words, txt_words[1:], txt_words[2:]):
                if w in self.used_gram_words and w2 in self.used_gram_words \
                        and w3 in self.used_gram_words:
                    features[f'contains({w}&{w2}&{w3})'] = 'Yes'

        if FeatureSetEnum.FOURGRAMS in fs_selection:
            if self.used_gram_words is None:
                raise exceptions.InsufficientDataException('Word set not defined.')
            for w, w2, w3, w4 in zip(txt_words, txt_words[1:], txt_words[2:], txt_words[3:]):
                if w in self.used_gram_words and w2 in self.used_gram_words \
                        and w3 in self.used_gram_words and w4 in self.used_gram_words:
                    features[f'contains({w}&{w2}&{w3}&{w4})'] = 'Yes'

        # MISC
        if FeatureSetEnum.REVIEWLEN in fs_selection:
            # features['@@@review_count']= 'A lot' if row['business']['review_count'] TODO add constant else 'A few'
            l = row['words']
            features['review_length'] = 'short' if l < 50 else 'middle' if l < 150 else 'long'
            features['review_length50'] = 'short' if l < 50 else 'middle'
            features['review_length100'] = 'short' if l < 100 else 'middle'
            features['review_length150'] = 'short' if l < 150 else 'middle'
            features['review_length35'] = 'short' if l < 35 else 'middle'
            features['review_length75'] = 'short' if l < 75 else 'middle'

        if FeatureSetEnum.SPELLCHECK in fs_selection:
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

        if FeatureSetEnum.COSINESIM in fs_selection:
            cos_sims = self.index[self.gensim_dictionary.doc2bow(self._tokenize(text))]
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
