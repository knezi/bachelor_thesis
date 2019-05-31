#!/bin/env python3
"""Creates classes for loading and storing data.

SampleTypeEnum - distinguishing between TRAIN, TEST, CROSSVALIDATION
Sample - container holding separately train,test,cv data
Data - class for loading, generating inferred and storing (in Sample) data
"""
from collections import defaultdict, Counter
from functools import reduce

from enum import Enum, unique, auto
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from matplotlib import pyplot
from sklearn.feature_selection import mutual_info_classif

from typing import List, Tuple, Dict, Set, Generator

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
from utils import top_n_indexes

FeatureDict = List[Tuple[Dict[str, any], str]]
"""Default format for storing dataset"""


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
    TFIDF = auto()
    ENTITIES = auto()


@unique
class LikeTypeEnum(Enum):
    """Enum used for denoting which class is classified"""
    USEFUL = 'useful'
    COOL = 'cool'
    FUNNY = 'funny'


class Data:
    """Load data from specified files to memory and make it accessible.

    You need to set max_tfidf and max_ngrams

    __init__ - take paths and load data to memory
    generate_sample - create a sample stored internally
TODO
    """
    # only words contained in this set will be used
    # when generating n-gram features
    used_ngrams: Set[str]
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
        self.used_ngrams = None
        self.used_entities = None
        self.tfidf: TfidfVectorizer = None
        self.max_tfidf = None
        self.max_ngrams = None

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

                dj['sentiment'] = gx3.sentiment.label if gx3.sentiment else 'n/a'
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

        self._regenerate_dictionaries()

        return train_size

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

    def get_raw_data(self, dataset: SampleTypeEnum, *attributes: str) \
            -> List[Tuple]:
        """Return raw data from specified dataset in a list of tuples.

        Each instance is a tuple of attributes specified in the argument in
        that order.

        :param dataset: returned dataset
        :param attributes: tuple of attributes as named in JSON
        :return: list of instances in the dataset
        """
        sample: List[Series] = self._sample.get_data(dataset)

        res: List[tuple] \
            = [(*self._filter_row(row, attributes),) for row in sample]

        return res

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
                         'sentiment', 'entities']] \
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

    def add_ngram(self, features: dict, tokens: List[str], n: int) -> None:
        """Add n-gram (specified in arg) into the given feature_dict.

        It counts only word appearing in self.used_ngrams

        :param features: already finished review
        :param tokens: tokens of the review
        :param n: `n`-grams
        """
        if self.used_ngrams is None:
            raise exceptions.InsufficientDataException('Word set not defined.')
        for i in range(len(tokens) + 1 - n):
            feature: str = 'contains('
            valid: bool = True
            for j in range(n):
                if tokens[i + j] in self.used_ngrams:
                    feature += tokens[i + j] + '&'
                else:
                    valid = False
                    break

            if valid:
                feature += ')'
                features[feature] = 'Yes'

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
        tokens = self._tokenize(text)
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
            self.add_ngram(features, tokens, 1)
        if FeatureSetEnum.BIGRAMS in fs_selection:
            self.add_ngram(features, tokens, 2)
        if FeatureSetEnum.TRIGRAMS in fs_selection:
            self.add_ngram(features, tokens, 3)
        if FeatureSetEnum.FOURGRAMS in fs_selection:
            self.add_ngram(features, tokens, 4)

        # TF-IDF
        if FeatureSetEnum.TFIDF in fs_selection:
            if self.tfidf is None:
                raise exceptions.InsufficientDataException('TF-IDF not initialized.')
            tfidf_vector = self.tfidf.transform([row.text]).toarray()[0]
            for fs, val in zip(self.tfidf.get_feature_names(), tfidf_vector):
                features[f'tf_idf({fs})'] = int(bool(val))

        # ENTITIES
        if FeatureSetEnum.ENTITIES in fs_selection:
            # we take all 1,2,3-grams and check if they're entities
            candidates: List \
                = list(map(lambda a: (a,), tokens)) \
                       + list(zip(tokens, tokens[1:])) \
                       + list(zip(tokens, tokens[1:], tokens[2:]))
            # entities are separated by space in standard form
            candidates_str: Generator[str] = map(" ".join, candidates)

            for ent in candidates_str:
                if ent in self.used_entities:
                    features[f'entity({ent})'] = 'Yes'

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

        return features

    def limit_train_size(self, size: int) -> None:
        """Directly call Sample.limit_train_size.

        :param size: size of train data
        """
        self._sample.limit_train_size(size)
        self._regenerate_dictionaries()

    def plot(self, data: DataGraph) -> None:
        """Plot given DataGraph.

        :param data: instance of DataGraph to be plotted"""
        self._statistics.plot(data)

    def _regenerate_dictionaries(self) -> None:
        """Regenerates used n-grams, tfidf everytime data change.

        This can occur either when the training size is changed or a new
        training set is obtained."""
        # TF-IDF
        tknz = nltk.TweetTokenizer()
        self.tfidf \
            = TfidfVectorizer(tokenizer=tknz.tokenize,
                              max_features=self.max_tfidf)
        # get_raw_data returns tuple of asked attributes (that is (text,))
        self.tfidf.fit(list(map(lambda a: a[0],
                                self.get_raw_data(SampleTypeEnum.TRAIN,
                                                  'text'))))

        # n-grams - mutual information
        vectorizer: CountVectorizer = CountVectorizer(tokenizer=tknz.tokenize)
        # get_raw_data returns tuple of asked attributes (that is (text,))
        word_matrix \
            = vectorizer.fit_transform(list(map(lambda i: i[0],
                                                self.get_raw_data(SampleTypeEnum.TRAIN,
                                                                  'text'))))
        labels: List[str] \
            = list(map(lambda a: a[0],
                       self.get_raw_data(SampleTypeEnum.TRAIN, 'classification')))

        mi = mutual_info_classif(word_matrix, labels)
        top_mi = top_n_indexes(mi, self.max_ngrams)
        ngrams = vectorizer.get_feature_names()
        self.used_ngrams = set(map(lambda i: ngrams[i], top_mi))

        # geneea entities
        # convert lists of entities into set and then join them into one set
        self.used_entities \
            = reduce(lambda a, b: a.union(b),
                     map(lambda i: set(i[0]),
                         self.get_raw_data(SampleTypeEnum.TRAIN,
                                           'entities')))

        # TODO restrict used entities
        # TODO statistics
        # print(self.used_ngrams)
        # pyplot.hist(mi)
        # pyplot.savefig('graphs/ngrams_histogram')
