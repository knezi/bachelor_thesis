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

            for ex in experiments['tasks']:
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
