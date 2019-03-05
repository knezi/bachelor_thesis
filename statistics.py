#!/bin/env python3
# TODO COMMENT
from collections import defaultdict

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from recordclass import RecordClass

import typing

from matplotlib import pyplot
import os


class Point(RecordClass):
    """Represent a point in a graph."""
    x: float
    y: float


class PointsPlot(RecordClass):
    """Store data points and format specs for a one type of data.

    points - list of RecordClasses Point
    fmt - format string for data plotting in mathplotilb"""
    points: typing.List[Point]
    fmt: str = ''


PointsPlots = typing.Dict[str, PointsPlot]
"""Typename for a dictionary holding several PointsPlots accesible by names.
Used for plotting graphs with more datalines.""" # TODO how to do this?


class Plot:
    """Wrapper class for plotting data.

    The constructor gets path to where store graphs.
    self.plot then plots given PointsPlots."""
    _fig: pyplot.figure
    _path: str

    def __init__(self, path: str) -> None:
        """Prepare the object for plotting.

        :param path: string path to the directory where graphs should be stored.
        """
        self._path = path
        # TODO resolution and stuff
        self._fig = pyplot.figure()

    def plot(self, data: PointsPlots, name: str, x_title: str = '',
             y_title: str = '', title: str = '') -> None:
        """ Plot `data` into a file 'name' stored in the given dir.

        :param data: data to be plotted
        :param name: filename of the graph
        :param x_title: label of x (optional)
        :param y_title:  label of y (optional)
        :param title: title of the graph (optional)
        """
        if title != '':
            self._fig.suptitle(title)

        # Each graph has exactly one axes for plotting
        self._fig.clf()
        ax: Axes = self._fig.subplots()

        # setting Axes
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        for label, pp in data.items():
            ax.plot(*zip(*pp.points), pp.fmt, label=label)
        ax.legend()

        self._fig.savefig(os.path.join(self._path, '{}.png'.format(name)))


class Statistics:
    """Class for buffering data points. Plot graphs and dump text from them."""
    _data: PointsPlots
    _file_prefix: str

    def __init__(self, path: str, file_prefix: str) -> None:
        """Set up paths & contruct default objects.

        :param path: directory of stored files
        :param file_prefix: file_prefix of all filenames
        """
        self._file_prefix = file_prefix #TODO why is this asking specify type?
        self._data = defaultdict(lambda: PointsPlot([], ''))
        self._plot = Plot(path)


    def add_points(self, x: float, value_dict: typing.Dict[str, float]) -> None:
        """Add points with the same x and various y for different types.

        :param x: x value same for all points
        :param value_dict: dictionary 'the name of the datatype' -> 'y value'
        """
        for key, val in value_dict.items():
            self._data[key].points.append(Point(x, val))

    def plot(self, name: str, keys: typing.List[str] = None) -> None:
        """Plot&text dump data of given types defined in a dictionary.

        :param name: filename of the graph
        :param keys: keys to the dictionary of types
        """
        fname: str= self._file_prefix + name
        if keys is None:
            keys = self._data.keys()

        self._plot.plot({key: self._data[key] for key in keys},
                        fname)

        # dump textual representation
        with open(os.path.join(self._path, fname + '.csv'), 'w') as w:
            w.write(';y;'.join(keys)+';y\n')

            for line in map(';'.join,
                            zip(*map(lambda x: [str(p.x) + ';' + str(p.y) for p in x],
                                [self._data[l].points for l in keys]))):
                w.write(line+'\n')

    def set_fmt(self, data_type: str, fmt: str) -> None:
        """Set fmt string for given type (default empty).

        :param data_type: type of data
        :param fmt: formatting directive for mathplotlib
        """
        self._data[data_type].fmt = fmt

    def clear_data(self):
        """Empty PointsPlots container."""
        self._data.clear()


if __name__ == '__main__':
    s = Statistics('graphs', 'P')

    s.add_points(1, {'hey': 2, 'a': 3})
    s.add_points(2, {'hey': 4, 'a': 3})

    s.set_fmt('hey', '')
    s.set_fmt('a', 'ro')

    s.plot('O')
    s.plot('O_hey', ['hey'])
