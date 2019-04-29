#!/bin/env python3
"""TODO comment"""
# classified = classifier.classify(fs)
import abc
import nltk
from typing import List, Tuple, Dict

from classifiers.baseclassifier import Classifier


class NaiveBayes(Classifier):

    def __init__(self, parameters: Dict) -> None:
        super().__init__(parameters)
        self.classifier = None

    def train(self, train_set: List[Tuple[Dict, str]]) -> None:
        """Train NaiveBayes."""
        super().train(train_set)
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

    @abc.abstractmethod
    def classify(self, instance: Dict) -> str:
        super().classify(instance)
        classified = self.classifier.classify(instance)
