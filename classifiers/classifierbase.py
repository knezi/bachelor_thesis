#!/bin/env python3
"""todo comment"""
from typing import Dict, Tuple, List
import abc


class ClassifierBase(metaclass=abc.ABCMeta):

    def __init__(self, parameters: Dict) -> None:
        """Set parameters of the classifier as defined in YAML file."""
        self.parameters: Dict = parameters

    @abc.abstractmethod
    def train(self, train_set: List[Tuple[Dict, str]]) -> None:
        """Train model with the given train_set.

        Can be used multiple times with different sets. The model will be
        always retrained from the beginning."""
        pass

    @abc.abstractmethod
    def classify(self, instance: Dict) -> str:
        # TODO throw not trained
        """Classify instance with the pretrained model stored in this class.."""
        return ""

