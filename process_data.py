#!/bin/env python3
# TODO COMMENT AUTHOR
import sys
from collections import defaultdict

import argparse
import yaml
from functools import reduce
from math import log2, ceil

from nltk.metrics import scores
from typing import DefaultDict, Dict, List, Tuple, Set

import classifiers
import preprocessors
from classifiers.classifierbase import ClassifierBase
from load_data import Data, SampleTypeEnum, FeatureSetEnum, LikeTypeEnum
from preprocessors.preprocessingbase import PreprocessorBase
from statistics import DataGraph


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
    negative_test: set = reduce(lambda a, b: a.union(b), testsets.values()) \
                         - testsets[evaluated_class.value]
    negative_ref: set = reduce(lambda a, b: a.union(b), refsets.values()) \
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
    clas_scores['accuracy'] = correctly_classified / len(data_set)

    return clas_scores


def main(config: argparse.Namespace) -> None:
    # TODO docstring
    with open(config.config_file, 'r') as cfg:
        experiments: dict = yaml.load(cfg)

    print('loading data')
    data = Data(config.yelp_file, config.geneea_file)

    print('generating samples')
    datasize: int = data.generate_sample(experiments['config']['chunks'],
                                         LikeTypeEnum.USEFUL)

    stats: DataGraph = DataGraph('', 'number of instances', 'percentage')

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
    while True:
        train_size: int \
            = int(datasize - datasize / experiments['config']['chunks'])
        train_size_log: int = int(ceil(log2(train_size)) + 1)
        for t_size in map(lambda x: min(2 ** x, train_size),
                          range(1, train_size_log)):
            print(f'SIZE {t_size}')

            data.max_tfidf = 10
            data.max_ngrams = 10
            data.limit_train_size(t_size)

            for ex in experiments['experiments']:
                # convert features to set:
                features: Set[FeatureSetEnum] \
                    = {FeatureSetEnum[f] for f in ex['features']}
                train_set = data.get_feature_dict(SampleTypeEnum.TRAIN, features,
                                                  ex['extra_data'])
                test_set = data.get_feature_dict(SampleTypeEnum.TEST, features,
                                                 ex['extra_data'])

                # preprocess data
                for pp in ex['preprocessing']:
                    prep: PreprocessorBase \
                        = getattr(preprocessors, pp).Preprocessor(ex['config'])
                    train_set = prep.process(train_set, SampleTypeEnum.TRAIN)
                    test_set = prep.process(test_set, SampleTypeEnum.TEST)

                cls: ClassifierBase \
                    = getattr(classifiers, ex['classificator']).Classifier(ex['config'])
                cls.train(train_set)

                evaluation: dict \
                    = compute_evaluation_scores(cls, test_set, LikeTypeEnum.USEFUL)

                stats.add_points(train_size, ex['name'], evaluation)

                # here needs to be done average agregation

        if not data.prepare_next_dataset():
            break

    # aggregate results here

    for g in experiments['graphs']:
        stats.name = g['name']
        stats.set_view(g['data'])
        data.plot(stats)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('config_file', type=str,
                           help='Config file in YAML.')
    argparser.add_argument('yelp_file', type=str,
                           help='Yelp data file.')
    argparser.add_argument('geneea_file', type=str,
                           help='Geneea data file.')

    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()

    main(argparser.parse_args(sys.argv[1:]))

    # pr.disable()
    # pr.print_stats(sort="calls")

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
