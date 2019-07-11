#!/bin/env python3
"""Define a baseline classifier."""
from scipy import sparse
from typing import Dict
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from classifiers.classifierbase import ClassifierBase


class Classifier(ClassifierBase):
    """DecisionTree

    This classifier requires matrix, not a dict."""

    def __init__(self, parameters: Dict) -> None:
        super().__init__(parameters)
        self._classifier = DecisionTreeClassifier()

    def train(self, train_set) -> None:
        """Train Decision Tree Classifier."""
        super().train(train_set)
        # split into data and target
        xlist, y = zip(*train_set)
        x = sparse.vstack(xlist)
        self._classifier.fit(x, y)

    def classify(self, instance) -> str:
        super().classify(instance)

        return self._classifier.predict(instance)[0]
