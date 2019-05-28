#!/bin/env python3
"""TODO comment"""
from collections import defaultdict

import unittest
from functools import reduce
from scipy import sparse

from typing import List, Tuple, Dict, Iterator, Set, Any

from load_data import SampleTypeEnum
from preprocessors.preprocessingbase import PreprocessorBase
from utils import Incrementer


class Preprocessor(PreprocessorBase):
    """TODO"""
    def __init__(self, parameters: Dict) -> None:
        super().__init__(parameters)
        self.feature_convert_table: Dict[str, Dict]
        self.all_fs: Tuple[str] = None

        def new_incremental_dict():
            """Return a dictionary where non-existing accessed value is len(dict)+1
            """
            inc: Incrementer = Incrementer()
            return defaultdict(inc)

        # this creates a two-dimensional dictionary
        # It is for converting feature values into integers.
        # Every time a new value of a feature is accessed, it is given
        # a unique int identifier
        # each Feature is incremented separately
        # 0 is left for missing values
        # usage: feature_convert_table[f1][v1] gives an int representation
        # of feature f1 with value v1, it is always the same
        self.feature_convert_table = defaultdict(new_incremental_dict)

    def process(self, dataset: List[Tuple], dataset_purpose: SampleTypeEnum) \
        -> List[Tuple[Any, str]]:
        """Return feature matrix, columns are attributes, rows instances.
        Last column is classification class.

        Attr values are converted with function _convert_feature_to_int.
        Matrix is represented as List [instance = List [ attr_value = Int] ]

        :param dataset:
        :param dataset_purpose:
        :return: (header, matrix)
                 header being tuple of string
                 matrix in the format specified above"""
        # each instance is tuple ({feature dict}, 'classification')
        super().process(dataset, dataset_purpose)

        if dataset_purpose == SampleTypeEnum.TRAIN:
            self.generate_keys(dataset)

        out_dataset: List[Tuple[Any, str]] = list()

        # iterating through instances
        for fs, cls in dataset:
            matrix_row: sparse\
                = sparse.lil_matrix((1, len(self.all_fs)), dtype=int)
            # iterating through features in the specified order
            for key_no, key in enumerate(self.all_fs):
                if key in fs:
                    matrix_row[0, key_no] \
                        = self.feature_convert_table[key_no][fs[key]]

            out_dataset.append((matrix_row, cls))

        return out_dataset

    def generate_keys(self, dataset) -> None:
        """Generate feature order and their mapping to the column in resulting
        matrix

        :param dataset:
        :return: None
        """
        all_keys: List[Set[str]] = [set(x[0].keys()) for x in dataset]
        # get all feature names
        # convert to tuple to preserve the order
        self.all_fs = sorted(tuple(reduce(lambda a, b: a.union(b), all_keys)))

    def get_fs(self) -> Tuple:
        """Return an order tuple of all features as their in the matrix"""
        return self.all_fs


class Matrix(unittest.TestCase):
    def test(self):
        prc = Preprocessor({})
        prc.process([({'a': 1, 'b': 2}, 'useful'),
                     ({'a': 1, 'c': 2}, 'useful'),
                     ], SampleTypeEnum.TRAIN)
        self.assertEqual(sorted(prc.get_fs()), sorted(('a', 'b', 'c')))

        for r1, r2 in zip(
            prc.process([({'a': 1, 'b': 2}, 'useful'),
                         ({'a': 1, 'c': 2}, 'useful'),
                         ({'a': 1, 'd': 2}, 'useful'),
                         ], SampleTypeEnum.TEST),
            [
                (sparse.csr_matrix([1, 1, 0]), 'useful'),
                (sparse.csr_matrix([1, 0, 1]), 'useful'),
                (sparse.csr_matrix([1, 0, 0]), 'useful'),
            ]):
            self.assertEqual(r1[1], r2[1])
            for el1, el2 in zip(r1[0].toarray()[0], r2[0].toarray()[0]):
                self.assertEqual(el1, el2)


if __name__ == "__main__":
    unittest.main()
