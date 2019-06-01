#!/bin/env python3
"""Define a wrapper for FastText as a preprocessor.
It simply writes data into files, which are than read by the fasttext classifier.

All data passed between this class and the fasttext classifier are ignored."""
from typing import List, Tuple, Dict, Any

from load_data import SampleTypeEnum
from preprocessors.preprocessingbase import PreprocessorBase


class Preprocessor(PreprocessorBase):
    """FastText wrapper - as a preprocessing it writes data into files.
    From this point any training data is useless.
    However, testing data will be used.
    It is only turned into string representation."""

    def __init__(self, parameters: Dict) -> None:
        super().__init__(parameters)
        self.path_prefix = parameters['path_prefix']

    def process(self, dataset, dataset_purpose: SampleTypeEnum) \
            -> List[Tuple[Any, str]]:
        """Create training files for fasttext from the current sample.

        Testing data are only turned into string representation.

        This must be the last preprocessing done.

        Beware, dataset must be in format List of instances where
            instance is a tuple (featuredict, classification, raw text)

        It makes use of self.dump_train and self.dump_test

        Train set is written into file {self.path_prefix}_train instance per line
        in format:
            __label__classification \
            features in the format _feature_value \
            text

        Test:
            It is not written anywhere, instead returned as a string.


        path_prefix is prefix used for all three files and is part of the
        config in YAML - namely experiment['config']['path_prefix']"""
        super().process(dataset, dataset_purpose)

        if dataset_purpose == SampleTypeEnum.TRAIN:
            self.dump_train(dataset)
        elif dataset_purpose == SampleTypeEnum.TEST:
            result: List[Tuple] = []
            for fs, cls, txt in dataset:
                result.append(
                    (Preprocessor.instance_representation((fs, txt)), cls))

            return result
        else:
            raise NotImplemented()

        return dataset

    def dump_train(self, dataset: List[Tuple]) -> None:
        """Create training files for fasttext.

        Train set is written into file {self.path_prefix}_train instance per line
        in format:
            __label__classification \
            features in the format _feature_value \
            text
        """
        train_p: str = f'{self.path_prefix}_train'

        with open(train_p, 'w') as train:
            # train set
            for (fs, clsf, txt) in dataset:
                print(f'__label__{clsf} '
                      f'{Preprocessor.instance_representation((fs, txt))}',
                      file=train)

    @staticmethod
    def instance_representation(instance: Tuple) -> str:
        """Return string of fasttext representation of an instance.

        :param instance:  (feature_dict, raw text)
        """
        # instance per line, newline is forbidden
        erase_nl_trans = str.maketrans({'\n': None})
        fs: Dict = instance[0]
        feature_string: str = ' '.join(map(lambda k: f'_{k}_{fs[k]}', fs))

        return f'{feature_string} {instance[1].translate(erase_nl_trans)}'
