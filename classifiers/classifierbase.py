#!/bin/env python3
"""todo comment"""
from typing import Dict, Tuple, List
import abc

from exceptions import NotTrainedException


class ClassifierBase(metaclass=abc.ABCMeta):

    def __init__(self, parameters: Dict) -> None:
        """Set parameters of the classifier as defined in YAML file."""
        self.parameters: Dict = parameters
        self.trained = False

    @abc.abstractmethod
    def train(self, train_set) -> None:
        """Train model with the given train_set.

        Can be used multiple times with different sets. The model will be
        always retrained from the beginning.

        type of train_set is dependent on the classifier and preprocessors"""
        self.trained = True

    @abc.abstractmethod
    def classify(self, instance) -> str:
        """Classify instance with the pretrained model stored in this class.."""
        if not self.trained:
            raise NotTrainedException("Classifier has not been trained yet.")
        return ""
