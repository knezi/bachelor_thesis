#!/bin/env python3
"""TODO comment"""
from subprocess import CompletedProcess
from typing import List, Tuple, Dict
import subprocess as sp
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier


from classifiers.classifierbase import ClassifierBase


class Classifier(ClassifierBase):
    """This classifier requires matrix, not a dict."""

    def __init__(self, parameters: Dict) -> None:
        super().__init__(parameters)
        if self.parameters['algorithm'] == 'one-R':
            self._classifier = DecisionTreeClassifier(max_depth=1)
        elif self.parameters['algorithm'] == 'zero-R':
            self._classifier = DummyClassifier(strategy='most_frequent')
        else:
            raise Exception(f'Unknown classifier {self.parameters["algorithm"]}')

    def train(self, train_set: List[Tuple[Dict, str]]) -> None:
        """Train Baseline Classifier."""
        super().train(train_set)
        # split into data and target
        X, y = zip(*train_set)
        self._classifier.fit(X, y)

    def classify(self, instance) -> str:
        super().classify(instance)

        return self._classifier.predict([instance])[0]
