#!/bin/env python3
"""Define Naive Bayes."""
import nltk
from typing import Dict

from classifiers.classifierbase import ClassifierBase


class Classifier(ClassifierBase):
    """Define Naive Bayes. No further parameters required.

    It expexct feature_dict."""


    def __init__(self, parameters: Dict) -> None:
        super().__init__(parameters)
        self._classifier = None

    def train(self, train_set) -> None:
        """Train NaiveBayes."""
        super().train(train_set)
        self._classifier = nltk.NaiveBayesClassifier.train(train_set)

    def classify(self, instance) -> str:
        super().classify(instance)
        return self._classifier.classify(instance)
