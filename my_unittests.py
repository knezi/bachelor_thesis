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
        sample.set_train(data)

        self.assertEqual(sample.get_train_basic()[0], ({'stars': 5}, 'useful'))
        self.assertEqual(sample.get_test_basic(), [])
        self.assertEqual(sample.get_crossvalidate_basic(), [])
        self.assertEqual(sample.get_train_basic(), sample.get_train_extended('classification'))
        self.assertEqual(sample.get_test_basic(), sample.get_test_extended('classification'))
        self.assertEqual(sample.get_crossvalidate_basic(), sample.get_crossvalidate_extended('classification'))

        test_data = \
            [({'stars': 3}, pd.Series({'a': 'd', 'classification': 'funny'})),
             ({'stars': 2}, pd.Series({'a': 'e', 'classification': 'not-funny'}))]
        sample.set_test(test_data)
        self.assertEqual(sample.get_test_extended('a')[1], ({'stars': 2}, 'e'))
        self.assertEqual(sample.get_test_extended()[1], ({'stars': 2},))

        sample.set_crossvalidation(data)
        self.assertEqual(sample.get_crossvalidate_extended('a', 'classification')[0], ({'stars': 3}, 'd', 'funny'))
        self.assertEqual(sample.get_train_basic()[0], ({'stars': 5}, 'useful'))

        self.assertEqual(sample.get_train_basic(), sample.get_train_extended('classification'))
        self.assertEqual(sample.get_test_basic(), sample.get_test_extended('classification'))
        self.assertEqual(sample.get_crossvalidate_basic(), sample.get_crossvalidate_extended('classification'))
        self.assertNotEqual(sample.get_train_basic(), sample.get_test_basic())


# TODO CLASS PLOT

if __name__ == "__main__":
    unittest.main()
