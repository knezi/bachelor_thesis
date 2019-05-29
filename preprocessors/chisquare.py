#!/bin/env python3
"""todo comment"""
from sklearn.feature_selection import chi2
import unittest
from typing import Dict, Tuple, List, Any, Set

from preprocessors import featureselectionbase, featurematrixconversion


class Preprocessor(featureselectionbase.Preprocessor):
    """Select most informative features with chi-square.
    """

    def evaluate_fs(self, matrix, labels) -> Any:
        """Abstract: define metrics for feature selection

        :returns: returns the actual evaluation"""
        return chi2(matrix, labels)[0]


class Chi2Test(unittest.TestCase):
    def test(self):
        pr = Preprocessor({'features_to_select': 1})
        pr.set_feature_restriction([
            ({'a': 1, 'b': 1}, 'useful'),
            ({'a': 1, 'b': 0}, 'useful'),
            ({'a': 0, 'b': 1}, 'not-useful'),
        ])
        self.assertSetEqual(pr._used_fs,
                         {'a'})


if __name__ == '__main__':
    unittest.main()
