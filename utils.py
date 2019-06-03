#!/bin/env python3
"""Define basic utils.

Incrementer - callable class returning last number + 1 when called
top_n_indexes - returns indexes of n highest/lowest elements.
get_abs_path - receives relative path to the root of the project
               and returns absolute path to it
"""
import os


class Incrementer:
    """Calling this function returns a number incremented by one per call.

    First returned number is 1, so the returned number is how many this
    instance has been already called."""

    def __init__(self):
        self.state: int = 0

    def __call__(self) -> int:
        self.state += 1
        return self.state


def top_n_indexes(data: list, n: int = None, reverse: bool = True) -> set:
    """Return set of n indexes with highest values in the list.
    :param data: list
    :param n: top n values
    :param reverse: if True (default) n hightest, n lowest otherwise
    """
    # sort by the number
    sort_list: list \
        = sorted(enumerate(data), key=lambda k: k[1], reverse=reverse)

    top: list = sort_list if n is None else sort_list[:n]
    return set(map(lambda a: a[0], top))


def get_abs_path(path: str) -> str:
    return os.path.join(os.path.dirname(__file__), path)


if __name__ == '__main__':
    assert (top_n_indexes([1, 8, 6, 4, 5], 3) == {1, 2, 4})
