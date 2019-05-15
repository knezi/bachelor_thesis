#!/bin/env python3
"""Class containing all created exceptions for the project."""

class DataMismatchException(Exception):
    """Raised when Geneea Data doesn't match those from Yelp."""
    pass

class InsufficientDataException(Exception):
    """Some part of data needed for analisys is missing."""
    pass

class NotTrainedException(Exception):
    """Some part of data needed for analisys is missing."""
    pass
