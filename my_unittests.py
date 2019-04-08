#!/bin/env python3


import pandas as pd
import unittest
import load_data
from statistics import DataGraph, Point, PointsPlot


class TestLoadData(unittest.TestCase):
    def test_sample(self):
        # create dummy data
        sample = load_data.Sample()
        data = \
            [({'stars': 5}, pd.Series({'a': 'b', 'classification': 'useful'})),
             ({'stars': 4}, pd.Series({'a': 'c', 'classification': 'not-useful'}))]
        sample.set_data(load_data.SampleTypeEnum.TRAIN, data)

        # testing all sets are properly set
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.TRAIN)[0],
                         ({'stars': 5}, 'useful'))
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.TEST),
                         [])
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.CROSSVALIDATION),
                         [])
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.TRAIN),
                         sample.get_data(load_data.SampleTypeEnum.TRAIN, 'classification'))
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.TEST),
                         sample.get_data(load_data.SampleTypeEnum.TEST, 'classification'))
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.CROSSVALIDATION),
                         sample.get_data(load_data.SampleTypeEnum.CROSSVALIDATION, 'classification'))

        # more dummy data
        test_data = \
            [({'stars': 3}, pd.Series({'a': 'd', 'classification': 'funny'})),
             ({'stars': 2}, pd.Series({'a': 'e', 'classification': 'not-funny'}))]
        sample.set_data(load_data.SampleTypeEnum.TEST, test_data)

        # testing test data has been set
        self.assertEqual(sample.get_data(load_data.SampleTypeEnum.TEST, 'a')[1],
                         ({'stars': 2}, 'e'))
        self.assertEqual(sample.get_data(load_data.SampleTypeEnum.TEST, )[1],
                         ({'stars': 2},))

        # Crossvalidation has been set
        sample.set_data(load_data.SampleTypeEnum.CROSSVALIDATION, test_data)
        self.assertEqual(sample.get_data(load_data.SampleTypeEnum.CROSSVALIDATION, 'a', 'classification')[0],
                         ({'stars': 3}, 'd', 'funny'))
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.TRAIN)[0],
                         ({'stars': 5}, 'useful'))

        # testing basic data vs extended
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.TRAIN),
                         sample.get_data(load_data.SampleTypeEnum.TRAIN, 'classification'))
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.TEST),
                         sample.get_data(load_data.SampleTypeEnum.TEST, 'classification'))
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.CROSSVALIDATION),
                         sample.get_data(load_data.SampleTypeEnum.CROSSVALIDATION, 'classification'))

        # test if data are not changed by sideeffects
        self.assertNotEqual(sample.get_data_basic(load_data.SampleTypeEnum.TRAIN),
                            sample.get_data_basic(load_data.SampleTypeEnum.TEST))

        self.assertRaises(IndexError, lambda: sample.limit_train_size(4))
        sample.limit_train_size(1)
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.TRAIN),
                         [({'stars': 5}, 'useful')])

    def test_load_data(self):
        # warning data has been tampered for testing purposes
        data = load_data.Data('unittests/data_unit.json', 'unittests/geneea_unit.json')
        # returns size of training set
        self.assertEqual(data.generate_sample(load_data.LikeTypeEnum.USEFUL, set()),
                         14)

        # test returned samples
        # empty features
        self.assertEqual(data.get_feature_dict(load_data.SampleTypeEnum.TRAIN)[0][0],
                         {})

        # only review len features
        self.assertEqual(data.generate_sample(load_data.LikeTypeEnum.USEFUL, {load_data.FeatureSetEnum.REVIEWLEN}),
                         14)
        self.assertEqual(len(data.get_feature_dict(load_data.SampleTypeEnum.TRAIN)[0][0]),
                         6)
        self.assertTrue('review_length' in data.get_feature_dict(load_data.SampleTypeEnum.TRAIN)[0][0])

        # number of instances
        self.assertEqual(len(data.get_feature_dict(load_data.SampleTypeEnum.TRAIN)),
                         14)
        data.limit_train_size(10)
        self.assertEqual(len(data.get_feature_dict(load_data.SampleTypeEnum.TRAIN)),
                         10)


class TestStatistics(unittest.TestCase):
    def test_data_graph(self):
        # create dummy data
        dg = DataGraph('name', 'xl', 'yl')
        dg.add_points(5, {'a': 2, 'b': 3})
        dg.clear_data()
        dg.add_points(1, {'a': 2, 'b': 3})
        dg.add_points(2, {'b': 3, 'd': 4})
        dg.add_points(2, {'c': 3})
        dg.set_view({'a', 'c', 'd'})
        dg.set_fmt('a', 'ro')

        # test string properties
        self.assertEqual(dg.name, 'name')
        self.assertEqual(dg.xlabel, 'xl')
        self.assertEqual(dg.ylabel, 'yl')

        # test data with the set view
        self.assertEqual(dg.get_data(),
                         {
                             'a': PointsPlot([Point(1, 2)], 'ro'),
                             'c': PointsPlot([Point(2, 3)], ''),
                             'd': PointsPlot([Point(2, 4)], '')
                         })

    def test_statistics(self):
        # for testing statistics, run statistics as main, it'll produce graphs
        pass


if __name__ == "__main__":
    unittest.main()
