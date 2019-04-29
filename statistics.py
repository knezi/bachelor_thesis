#!/bin/env python3
"""Creates helpers classes for plotting graphs and dumping textual data.

Point - record class of a single point
DataLine - record class for a list of types Point
DataLines - store named instances of DataLine
DataGraph - accept data as it goes from the programme flow, aggregate it and
            can be passed as a whole for plotting
Statistics - prepares directories for graphs and plots DataGraphs
"""
from collections import defaultdict

from typing import List, Dict, Generator, Set

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from recordclass import RecordClass

from matplotlib import pyplot
import os


class Point(RecordClass):
    """Represent a point in a graph."""
    x: float
    y: float


class DataLine(RecordClass):
    """Store data points and format specs for a one type of data.

    points - list of RecordClasses Point
    fmt - format string for data plotting in mathplotilb"""
    points: List[Point]
    fmt: str = ''


DataLines = Dict[str, DataLine]
"""Typename for a dictionary holding several DataLine_s accesible by names.
Used for plotting graphs with more datalines.""" # TODO how to do this?


class DataGraph:
    """Class for buffering data points and other graph related properties.

        DataGraph - accept data as it goes from the programme flow, aggregate it
        and can be passed to Statistics as a whole for plotting"""
    _keys: Set
    _name: str
    _data: DataLines

    def __init__(self, name: str = '', xlabel: str = '', ylabel: str = '') -> None:
        """Init empty data object.

        :param name: name.{png,csv} will be used for graph&csv dump respectively
        :param xlabel: label of x axis
        :param ylabel: label of y axis
        """
        self._data = defaultdict(lambda: DataLine([], ''))
        self.name = name
        self.xlabel = xlabel
        self.ylabel = ylabel
        self._keys = None

    def add_points(self, x: float, value_dict: Dict[str, float]) -> None:
        """Add points with the same x and various y for different types.

        Y values None are replaced with dummy -1

        :param x: x value same for all points
        :param value_dict: dictionary 'the name of the datatype' -> 'y value'
        """
        for key, val in value_dict.items():
            if val is None:
                val = -1
            self._data[key].points.append(Point(x, val))

    def set_fmt(self, data_type: str, fmt: str) -> None:
        """Set fmt string for given type (default empty).

        :param data_type: type of data
        :param fmt: formatting directive for mathplotlib
        """
        self._data[data_type].fmt = fmt

    def clear_data(self):
        """Empty DataLines container."""
        self._data.clear()

    def set_view(self, keys: Set) -> None:
        """Set keys visible when dumping data with get_data.

        :param keys: names of data types"""
        self._keys = keys

    def get_data(self) -> DataLines:
        """Returns DataLines of data from restricted view.

        The view is set by self.restrict_view"""

        if self._keys is not None:
            return {key: self._data[key] for key in self._keys}
        return self._data


class Statistics:
    """Plot graphs and dump text from them.

    Prepares directories for graphs and plots DataGraphs.
    The constructor gets path to where store graphs.
    Otherwise, stores no data, only immediately plots arguments of functions.
    """
    _file_prefix: str
    _path: str

    def __init__(self, path: str, file_prefix: str = '') -> None:
        """Set up paths & contruct default objects.

        :param path: directory of stored files
        :param file_prefix: file_prefix of all filenames
        """
        self._path = path
        self._file_prefix = file_prefix

        # TODO resolution and stuff
        self._fig = pyplot.figure()

    def set_file_prefix(self, prefix: str) -> None:
        """Each generated file will be prefixed with prefix

        :param prefix: prefix of each file"""
        self._file_prefix = prefix

    def plot(self, data: DataGraph) -> None:
        """Plot&text dump data of given types defined in a dictionary.

        :param data: plotted data"""
        fname: str= self._file_prefix + data.name

        # Each graph has exactly one axes for plotting
        self._fig.clf()
        ax: Axes = self._fig.subplots()

        # setting Axes
        ax.set_xlabel(data.xlabel)
        ax.set_ylabel(data.ylabel)
        pps: DataLines = data.get_data()
        for label, pp in pps.items():
            # convert data from [(x1,y1),...] to [[x1,x2...], [y1, y2...]]
            ax.plot(*zip(*pp.points), pp.fmt, label=label)
        ax.legend()

        self._fig.savefig(os.path.join(self._path, f'{fname}.png'))

        # dump textual representation in CSV, columns being:
        # property1,x;prop1.y;...;propn.x;propn.y
        # convert keys to tuple to preserve the order
        keys = tuple(pps.keys())
        with open(os.path.join(self._path, f'{fname}.csv'), 'w') as w:
            # header
            w.write(';y;'.join(keys)+';y\n')

            # convert data into iterable of lines being list of strings
            # and round values to three digits
            # zip(*iterable) reverses columns and rows
            data_lists: Generator[List[str], None, None] \
                = zip(*map(lambda x: [f'{round(p.x, 3)};{round(p.y, 3)}' for p in x],
                           [pps[k].points for k in keys]))
            # convert data into lines (str)
            data_lines: Generator[str, None, None] = map(';'.join, data_lists)

            for line in data_lines:
                w.write(line+'\n')


# only for testing purposes
if __name__ == '__main__':
    s = Statistics('graphs', 'P')

    dg = DataGraph()
    dg.add_points(1, {'hey': 2, 'a': 3})
    dg.add_points(2, {'hey': 4, 'a': 3})

    dg.set_fmt('hey', '')
    dg.set_fmt('a', 'ro')

    dg.set_name('trial')
    s.plot(dg)
    dg.set_name('trial2')
    dg.set_view(['hey'])
    s.plot(dg)
