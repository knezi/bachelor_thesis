#!/bin/env python3
"""Define fasttext classifier."""
from subprocess import CompletedProcess
from typing import Dict
import subprocess as sp

from classifiers.classifierbase import ClassifierBase


class Classifier(ClassifierBase):
    """Define FastText.

    It expects data to be written in files as defined by config. For training
    it simply reads file path_prefix_train and creates path_prefix_model
    For testing then uses the instances given to the output passing
    them automatically to stdin of fasttext.

    It expects three parameters:
        path_prefix - the prefix to path to all needed files
        """

    def __init__(self, parameters: Dict) -> None:
        super().__init__(parameters)
        self.path_prefix = self.parameters['path_prefix']
        self.executable = self.parameters['executable']
        self.config_ft = self.parameters['config_ft']

    def train(self, train_set) -> None:
        """Train FastText model."""
        super().train(train_set)

        finished_process: CompletedProcess \
            = sp.run([self.executable,
                      'supervised',
                      '-input', f'{self.path_prefix}_train',
                      '-output', f'{self.path_prefix}_model',
                      *self.config_ft],
                     encoding='utf-8')

        if finished_process.returncode != 0:
            raise Exception(f'fasttext failed.')

    def classify(self, instance) -> str:
        super().classify(instance)

        finished_process: CompletedProcess \
            = sp.run([self.executable,
                      'predict',
                      f'{self.path_prefix}_model.bin',
                      '-'],
                     input=instance,
                     encoding='utf-8',
                     stdout=sp.PIPE)

        # strip __label__ -> first niene characters
        return finished_process.stdout.strip()[9:]
