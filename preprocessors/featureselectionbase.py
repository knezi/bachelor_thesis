#!/bin/env python3
"""todo comment"""
from typing import Dict, Tuple, List, Any, Set
import abc

from load_data import SampleTypeEnum
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
        self.used_fs: Set = None

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
                ({k: v for k, v in row[0].items() if k in self.used_fs},
                 row[1])
            )

        return dataset_out

    @abc.abstractmethod
    def set_feature_restriction(self, dataset) -> None:
        """Set self.used_fs to features that will only be used.

        This method must be defined in a child class."""
        pass
