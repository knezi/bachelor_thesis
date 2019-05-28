#!/bin/env python3
"""todo comment"""
from pandas._libs import sparse
from scipy import sparse
import unittest
from typing import Dict, Tuple, List, Any, Set, Generator
import abc

from load_data import SampleTypeEnum
from preprocessors import featurematrixconversion
from preprocessors.preprocessingbase import PreprocessorBase


class Preprocessor(PreprocessorBase):
    """Convert feature_dict into another feature_dict with restricted features.

    Servers as a baseclass for different filters and wrappers
    for feature selection.
    """

    def __init__(self, parameters: Dict) -> None:
        """Set parameters of the classifier as defined in YAML file."""
        super().__init__(parameters)
        self.parameters: Dict = parameters
        self.select_fs: int = self.parameters['features_to_select']
        self._used_fs: Set = None

    def process(self, dataset, dataset_purpose: SampleTypeEnum) \
            -> List[Tuple[Any, str]]:
        """Filter taking data as returned from get_feature_dict
        and return changed data in the same format.

        Beware this function is used for both training and testing data. It
        must not leak the actual classification label in the resulting data.

        :param dataset_purpose: for what the dataset will be used,
            it is guaranteed that always first training_set will be passed,
            followed by arbitrary number of test and crossvalidation sets
        :param dataset: in dataset
        :returns: out dataset
        """
        if dataset_purpose == SampleTypeEnum.TRAIN:
            self.set_feature_restriction(dataset)

        dataset_out: List[Tuple[Any, str]] = []

        # restrict dataset to only used features
        for row in dataset:
            dataset_out.append(
                ({k: v for k, v in row[0].items() if k in self._used_fs},
                 row[1])
            )

        return dataset_out

    def set_feature_restriction(self, dataset) -> None:
        """Set self.used_fs to features that will only be used.
        
        It will call evalute_fs(matrix, labels) and set top
        self.select_fs features into self.used_fs.
        """
        matrix, labels, all_fs = self.convert_to_matrix(dataset)

        eval_vals = self.evaluate_fs(sparse.vstack(matrix), labels)

        # we enumerate it
        # sort by eval values
        # drop vales
        # to have a generator of top indexes
        self._used_fs = set()
        fs_indexes: Generator[int] \
            = map(lambda k: k[0],
                  sorted(enumerate(eval_vals),
                         key=lambda el: el[1],
                         reverse=True)[:self.select_fs])

        for i in fs_indexes:
            self._used_fs.add(all_fs[i])

    @abc.abstractmethod
    def evaluate_fs(self, matrix, labels) -> Any:
        """Abstract: define metrics for feature selection

        :returns: returns the actual evaluation"""
        pass

    @staticmethod
    def convert_to_matrix(dataset) -> Tuple[list, list, Tuple[str]]:
        """Convert dataset into matrix and label. Most filters and wrappers
        need this.
        :param dataset:
        :return: returns matrix, labels, names of features
        """
        prep: featurematrixconversion.Preprocessor \
            = featurematrixconversion.Preprocessor({})
        dataset_matrix: List[Tuple[Any, str]] \
            = prep.process(dataset, SampleTypeEnum.TRAIN)
        data: Tuple[list, list] = tuple(zip(*dataset_matrix))
        return data + (prep.all_fs,)


class Matrix(unittest.TestCase):
    def test(self):
        for r1, r2 in zip(
                Preprocessor.convert_to_matrix([({'a': 1, 'b': 2}, 'useful'),
                                                ({'a': 1, 'c': 2}, 'useful'),
                                                ({'a': 1, 'd': 2}, 'useful'),
                                                ])[0],
                [
                    sparse.csr_matrix([1, 1, 0]),
                    sparse.csr_matrix([1, 0, 1]),
                    sparse.csr_matrix([1, 0, 0]),
                ]):
            for el1, el2 in zip(r1.toarray()[0], r2.toarray()[0]):
                self.assertEqual(el1, el2)


if __name__ == "__main__":
    unittest.main()
