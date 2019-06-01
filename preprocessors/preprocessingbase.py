#!/bin/env python3
"""File defining preprocessing base which provides abstract class for all
preprocessing done on data before passed to classifier."""
from typing import Dict, Tuple, List, Any
import abc

from load_data import SampleTypeEnum


class PreprocessorBase(metaclass=abc.ABCMeta):
    """Base class for all preprocessors.

    It defines methods process which acts as a UNIX filter. It gets data
    and processed returns them.

    It is also given the purpose of data (training vs testing), it is expected
    the class may differentiate behaviour on this
    (e.g. for training fit_transform, for testing only transform).
    """

    def __init__(self, parameters: Dict) -> None:
        """Set parameters of the classifier as defined in YAML file."""
        self.parameters: Dict = parameters

    @abc.abstractmethod
    def process(self, dataset, dataset_purpose: SampleTypeEnum)\
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
        pass

