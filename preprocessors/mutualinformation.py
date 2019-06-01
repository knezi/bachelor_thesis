#!/bin/env python3
"""Define preprocessor for mutual information - it selects only features
with highest mutual information with the class"""
from sklearn.feature_selection import chi2, mutual_info_classif
import unittest
from typing import Dict, Tuple, List, Any, Set

from preprocessors import featureselectionbase, featurematrixconversion


class Preprocessor(featureselectionbase.Preprocessor):
    """Select most informative features with mutual information.
    """

    def evaluate_fs(self, matrix, labels) -> Any:
        """Abstract: define metrics for feature selection

        :returns: returns the actual evaluation"""
        return mutual_info_classif(matrix, labels)


class MITest(unittest.TestCase):
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
