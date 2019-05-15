#!/bin/env python3
"""todo comment"""
from typing import Dict, Tuple, List
import abc

from load_data import SampleTypeEnum


class PreprocessorBase(metaclass=abc.ABCMeta):

    def __init__(self, parameters: Dict) -> None:
        """Set parameters of the classifier as defined in YAML file."""
        self.parameters: Dict = parameters

    @abc.abstractmethod
    def process(self, dataset: List[Tuple], dataset_purpose: SampleTypeEnum)\
            ->  List[Tuple]:
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
        pass

