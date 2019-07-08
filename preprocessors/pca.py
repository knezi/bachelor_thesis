#!/bin/env python3
"""Define preprocessor for mutual information - it selects only features
with highest mutual information with the class

It automatically converts data into a matrix and keeps track of features used.
And then back to feature_dict (as required by nltk.naive_bayes
"""
from pandas._libs import sparse
from scipy import sparse
import unittest

from sklearn.decomposition import TruncatedSVD
from typing import Dict, Tuple, List, Any, Set, Generator
import abc

from load_data import SampleTypeEnum
from preprocessors import featurematrixconversion
from preprocessors.preprocessingbase import PreprocessorBase


class Preprocessor(PreprocessorBase):
    """Convert feature_dict into another feature_dict
    """

    def __init__(self, parameters: Dict) -> None:
        """Set parameters of the classifier as defined in YAML file."""
        super().__init__(parameters)
        self.parameters: Dict = parameters
        self.select_fs: int = self.parameters['features_to_select']
        self._pca: TruncatedSVD = TruncatedSVD(n_components=self.select_fs)
        self._mtrx_conv: featurematrixconversion.Preprocessor \
            = featurematrixconversion.Preprocessor({})

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
        dataset: List[Tuple[Any, str]] \
            = self._mtrx_conv.process(dataset, dataset_purpose)
        data, labels = tuple(zip(*dataset))
        dataset_matrix = sparse.vstack(data)

        if dataset_purpose == SampleTypeEnum.TRAIN:
            reduced_matrix = self._pca.fit_transform(dataset_matrix)
        else:
            reduced_matrix = self._pca.transform(dataset_matrix)



        dataset_out: List[Tuple[Any, str]] = []

        for fs,lbl in zip(reduced_matrix, labels):
            fs_dict: Dict = {}
            for i, val in enumerate(fs):
                fs_dict[i] = val
            dataset_out.append((fs_dict, lbl))

        return dataset_out


if __name__ == "__main__":
    a = Preprocessor({'features_to_select': 2})
    b = a.process( [
        ({1: 2, 2: 2}, 'a'),
        ({2: 1, 3: 5}, 'b'),
        ({1: 1, 2: 9}, 'a'),
    ], SampleTypeEnum.TRAIN)
    print(b)

    b = a.process( [
        ({1: 2, 2: 2}, 'a'),
        ({2: 2, 4: 4}, 'b'),
        ({1: 1, 2: 9}, 'a'),
    ], SampleTypeEnum.TEST)
    print(b)
