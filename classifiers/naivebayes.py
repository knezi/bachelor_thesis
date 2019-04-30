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
        self._classifier = None

    def train(self, train_set: List[Tuple[Dict, str]]) -> None:
        """Train NaiveBayes."""
        super().train(train_set)
        self._classifier = nltk.NaiveBayesClassifier.train(train_set)

    def classify(self, instance: Dict) -> str:
        super().classify(instance)
        return self._classifier.classify(instance)
