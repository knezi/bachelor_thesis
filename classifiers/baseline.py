#!/bin/env python3
"""Define a baseline classifier."""
from scipy import sparse
from typing import Dict
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from classifiers.classifierbase import ClassifierBase


class Classifier(ClassifierBase):
    """Baseline classifier - the type must be passed in config 'algorithm'
    possible values are:
    'one-R'
    'zero-R'

    This classifier requires matrix, not a dict."""

    def __init__(self, parameters: Dict) -> None:
        super().__init__(parameters)
        if self.parameters['algorithm'] == 'one-R':
            self._classifier = DecisionTreeClassifier(max_depth=1)
        elif self.parameters['algorithm'] == 'zero-R':
            self._classifier = DummyClassifier(strategy='most_frequent')
        else:
            raise Exception(f'Unknown classifier {self.parameters["algorithm"]}')

    def train(self, train_set) -> None:
        """Train Baseline Classifier."""
        super().train(train_set)
        # split into data and target
        xlist, y = zip(*train_set)
        x = sparse.vstack(xlist)
        self._classifier.fit(x, y)

    def classify(self, instance) -> str:
        super().classify(instance)

        return self._classifier.predict(instance)[0]
