#!/bin/env python3
# TODO

class Incrementer:
    """Calling this function returns a number incremented by one per call.

    First returned number is 1, so the returned number is how many this
    instance has been already called."""

    def __init__(self):
        self.state: int = 0

    def __call__(self) -> int:
        self.state += 1
        return self.state
