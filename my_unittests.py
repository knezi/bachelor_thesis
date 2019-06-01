#!/bin/env python3

import pandas as pd
import unittest

import classifiers
import exceptions
import load_data
from process_data import compute_evaluation_scores
from statistics import DataGraph, Point, DataLine


class TestLoadData(unittest.TestCase):
    def test_sample(self):
        # create dummy data
        sample = load_data.Sample()
        data = \
            [pd.Series({'a': 'b', 'classification': 'useful'}),
             pd.Series({'a': 'c', 'classification': 'not-useful'})]
        sample.add_chunk([data[0]])
        sample.add_chunk([data[1]])
        self.assertEqual(
            sample.get_data(load_data.SampleTypeEnum.TRAIN),
            None
        )

        # testing all sets are properly set
        sample.start_iter()
        self.assertTrue(sample.get_data(load_data.SampleTypeEnum.TEST)[0]
                        .equals(pd.Series({'a': 'b', 'classification': 'useful'})))
        self.assertTrue(sample.get_data(load_data.SampleTypeEnum.TRAIN)[0]
                        .equals(pd.Series({'a': 'c', 'classification': 'not-useful'})))

        # more dummy data
        data = \
            [pd.Series({'a': 'd', 'classification': 'funny'}),
             pd.Series({'a': 'e', 'classification': 'not-funny'})]
        sample.add_chunk([data[0]])
        sample.add_chunk([data[1]])

        sample.start_iter()
        self.assertTrue(sample.get_data(load_data.SampleTypeEnum.TEST)[0]
                        .equals(pd.Series({'a': 'b', 'classification': 'useful'})))
        self.assertTrue(sample.get_data(load_data.SampleTypeEnum.TRAIN)[0]
                        .equals(pd.Series({'a': 'c', 'classification': 'not-useful'})))
        self.assertEqual(
            len(sample.get_data(load_data.SampleTypeEnum.TRAIN)),
            3)

        sample.next_iter()
        self.assertTrue(sample.get_data(load_data.SampleTypeEnum.TEST)[0]
                        .equals(pd.Series({'a': 'c', 'classification': 'not-useful'})))
        self.assertTrue(sample.get_data(load_data.SampleTypeEnum.TRAIN)[0]
                        .equals(pd.Series({'a': 'b', 'classification': 'useful'})))
        self.assertEqual(
            len(sample.get_data(load_data.SampleTypeEnum.TRAIN)),
            3)

        # test if data are not changed by sideeffects
        identical = True
        for a, b in zip(sample.get_data(load_data.SampleTypeEnum.TRAIN),
                        sample.get_data(load_data.SampleTypeEnum.TEST)):
            if not a.equals(b):
                identical = False
                break
        self.assertFalse(identical)

        self.assertRaises(IndexError, lambda: sample.limit_train_size(4))
        sample.limit_train_size(1)
        self.assertTrue(len(sample.get_data(load_data.SampleTypeEnum.TRAIN)) == 1)

        self.assertEqual(sample.next_iter(), True)
        self.assertEqual(sample.next_iter(), True)
        self.assertEqual(sample.next_iter(), False)

    def test_load_data(self):
        # warning data has been tampered for testing purposes
        self.assertRaises(exceptions.DataMismatchException,
                          lambda: load_data.Data('unittests/data_unit_mismatch.json',
                                                 'unittests/geneea_unit_mismatch.json'))

        # warning data has been tampered for testing purposes
        data = load_data.Data('unittests/data_unit.json', 'unittests/geneea_unit.json')
        # returns size of training set
        self.assertEqual(data.generate_sample(load_data.LikeTypeEnum.USEFUL),
                         14)

        # test returned samples
        # empty features
        self.assertEqual(data.get_feature_dict(load_data.SampleTypeEnum.TRAIN, set())[0][0],
                         {})

        # only review len features
        self.assertEqual(data.generate_sample(load_data.LikeTypeEnum.USEFUL),
                         14)
        self.assertEqual(len(data.get_feature_dict(load_data.SampleTypeEnum.TRAIN, set())),
                         14)
        # it's 6 + 7th is classification
        # header
        # TODO this is gone
        # self.assertEqual(len(data.get_feature_matrix(load_data.SampleTypeEnum.TRAIN)[0]),
        #                  7)
        # # instance
        # self.assertEqual(len(data.get_feature_matrix(load_data.SampleTypeEnum.TRAIN)[1][1]),
        #                  7)
        # it's 1 feature only - classification
        # self.assertEqual(len(data.get_raw_data(load_data.SampleTypeEnum.TRAIN, 'classification')[0]),
        #                  1)
        # cls = data.get_raw_data(load_data.SampleTypeEnum.TRAIN, 'classification')[1]
        # test that we really got classification, not some other attribute
        # self.assertTrue(cls[0] == 'useful' or cls[0] == 'not-useful')
        self.assertTrue('review_length' in
                        data.get_feature_dict(load_data.SampleTypeEnum.TRAIN, {load_data.FeatureSetEnum.REVIEWLEN})[0][
                            0])

        # number of instances
        self.assertEqual(len(data.get_feature_dict(load_data.SampleTypeEnum.TRAIN, set())),
                         14)
        data.limit_train_size(10)
        self.assertEqual(len(data.get_feature_dict(load_data.SampleTypeEnum.TRAIN, set())),
                         10)

        # test insufficient data exception
        data.generate_sample(load_data.LikeTypeEnum.USEFUL)
        self.assertEqual(data.generate_sample(load_data.LikeTypeEnum.USEFUL),
                         14)

        # test add n-grams
        data.used_ngrams = {'a', 'b'}
        fs = {'c': 2}
        data.add_ngram(fs, ['b', 'b', 'c', 'a'], 2)
        self.assertEqual(fs,
                         {'c': 2,
                          'contains(b&b&)': 'Yes',
                          })


class TestStatistics(unittest.TestCase):
    def test_data_graph(self):
        # create dummy data
        dg = DataGraph('name', 'xl', 'yl')
        dg.add_points(5, 'n', {'a': 2, 'b': 3})
        dg.clear_data()
        dg.add_points(1, 'n', {'a': 2, 'b': 3})
        dg.add_points(2, 'n', {'b': 3, 'd': 4})
        dg.add_points(2, 'n', {'c': 3})
        dg.set_view({'n': {'a', 'c', 'd'}})
        dg.set_fmt('n', 'a', 'ro')

        # test string properties
        self.assertEqual(dg.name, 'name')
        self.assertEqual(dg.xlabel, 'xl')
        self.assertEqual(dg.ylabel, 'yl')

        # test data with the set view
        self.assertEqual(dg.get_data(),
                         {'n': {
                             'a': DataLine([Point(1, 2)], 'ro'),
                             'c': DataLine([Point(2, 3)], ''),
                             'd': DataLine([Point(2, 4)], '')
                         }})

    def test_statistics(self):
        # for testing statistics, run statistics as main, it'll produce graphs
        pass


class TestClassifier(unittest.TestCase):
    def test_naive_bayes(self):
        nb = classifiers.naivebayes.Classifier({})
        nb.train([({'a': 1, 'b': 1}, 'a'),
                  ({'a': 1, 'b': 2}, 'b')])

        self.assertEqual(nb.classify({'a': 1, 'b': 1}), 'a')
        self.assertEqual(nb.classify({'a': 1, 'b': 2}), 'b')


class TestProcessData(unittest.TestCase):
    def test_evaluation(self):
        nb = classifiers.naivebayes.Classifier({})
        nb.train([({'a': 1, 'b': 1}, 'useful'),
                  ({'a': 1, 'b': 2}, 'not-useful'),
                  ({'a': 1, 'b': 3}, 'not-useful')])

        test_set = [({'a': 1, 'b': 1}, 'useful'),  # TP
                    ({'a': 1, 'b': 2}, 'not-useful'),  # TN
                    ({'a': 1, 'b': 2}, 'not-useful'),  # TN
                    ({'a': 1, 'b': 3}, 'useful'),  # FN
                    ({'a': 1, 'b': 3}, 'useful'),  # FN
                    ({'a': 1, 'b': 3}, 'useful'),  # FN
                    ({'a': 1, 'b': 1}, 'not-useful'),  # FP
                    ({'a': 1, 'b': 1}, 'not-useful'),  # FP
                    ({'a': 1, 'b': 1}, 'not-useful'),  # FP
                    ({'a': 1, 'b': 1}, 'not-useful')]  # FP
        # TP 1
        # TN 2
        # FP 4
        # FN 3

        metrics: dict = compute_evaluation_scores(nb, test_set, load_data.LikeTypeEnum.USEFUL)
        expected_out: dict = {'accuracy': 3 / 10,
                              'precision': 1 / (1 + 4),
                              'recall': 1 / (1 + 3),
                              'f_measure': 2 * 1 / 5 * 1 / 4 / (1 / 5 + 1 / 4),
                              'tp': 1 / 10,
                              'tn': 2 / 10,
                              'fp': 4 / 10,
                              'fn': 3 / 10}

        for k in expected_out.keys():
            self.assertEqual(round(metrics[k], 3),
                             round(expected_out[k], 3),
                             msg=f'metrics {k} failed.')


if __name__ == "__main__":
    unittest.main()
