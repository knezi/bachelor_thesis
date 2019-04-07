#!/bin/env python3


import pandas as pd
import unittest
import load_data


class TestLoadData(unittest.TestCase):
    def test_sample(self):
        sample = load_data.Sample()
        data = \
            [({'stars': 5}, pd.Series({'a': 'b', 'classification': 'useful'})),
             ({'stars': 4}, pd.Series({'a': 'c', 'classification': 'not-useful'}))]
        sample.set_data(load_data.SampleTypeEnum.TRAIN, data)

        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.TRAIN)[0], ({'stars': 5}, 'useful'))
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.TEST), [])
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.CROSSVALIDATION), [])
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.TRAIN), sample.get_data(load_data.SampleTypeEnum.TRAIN, 'classification'))
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.TEST), sample.get_data(load_data.SampleTypeEnum.TEST, 'classification'))
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.CROSSVALIDATION), sample.get_data(load_data.SampleTypeEnum.CROSSVALIDATION, 'classification'))

        test_data = \
            [({'stars': 3}, pd.Series({'a': 'd', 'classification': 'funny'})),
             ({'stars': 2}, pd.Series({'a': 'e', 'classification': 'not-funny'}))]
        sample.set_data(load_data.SampleTypeEnum.TEST, test_data)
        self.assertEqual(sample.get_data(load_data.SampleTypeEnum.TEST, 'a')[1], ({'stars': 2}, 'e'))
        self.assertEqual(sample.get_data(load_data.SampleTypeEnum.TEST, )[1], ({'stars': 2},))

        sample.set_data(load_data.SampleTypeEnum.CROSSVALIDATION, test_data)
        self.assertEqual(sample.get_data(load_data.SampleTypeEnum.CROSSVALIDATION, 'a', 'classification')[0], ({'stars': 3}, 'd', 'funny'))
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.TRAIN)[0], ({'stars': 5}, 'useful'))

        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.TRAIN), sample.get_data(load_data.SampleTypeEnum.TRAIN, 'classification'))
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.TEST), sample.get_data(load_data.SampleTypeEnum.TEST, 'classification'))
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.CROSSVALIDATION), sample.get_data(load_data.SampleTypeEnum.CROSSVALIDATION, 'classification'))

        self.assertNotEqual(sample.get_data_basic(load_data.SampleTypeEnum.TRAIN), sample.get_data_basic(load_data.SampleTypeEnum.TEST))

        self.assertRaises(IndexError, lambda: sample.limit_train_size(4))
        sample.limit_train_size(1)
        self.assertEqual(sample.get_data_basic(load_data.SampleTypeEnum.TRAIN), [({'stars': 5}, 'useful')])

    def test_load_data(self):
        # warning data has been tampered for testing purposes
        data = load_data.Data('unittests/data_unit.json', 'unittests/geneea_unit.json')
        # returns size of training set
        self.assertEqual(data.generate_sample('useful', {}), 14)
        self.assertEqual(data.get_feature_dict(load_data.SampleTypeEnum.TRAIN)[0][0], {})
        self.assertEqual(data.generate_sample('useful', {load_data.FeatureSet.REVIEWLEN}), 14)
        self.assertEqual(len(data.get_feature_dict(load_data.SampleTypeEnum.TRAIN)[0][0]), 6)
        self.assertTrue('review_length' in data.get_feature_dict(load_data.SampleTypeEnum.TRAIN)[0][0])


class TestStatistics(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
