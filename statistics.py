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
    x: float
    y: float


class PointsPlot(RecordClass):
    points: typing.List[Point]
    fmt: str = ''


PointsPlots = typing.Dict[str, PointsPlot]


class Plot:
    fig: Figure

    def __init__(self, path):
        self.path = path
        # TODO resolution and stuff
        self.fig = pyplot.figure()

    def plot(self, data: PointsPlots, name: str, x_title: str = '',
             y_title: str = '', title: str = '') -> None:
        self.fig.clf()

        if title != '':
            self.fig.suptitle(title)

        ax: Axes = self.fig.subplots()
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        for label, pp in data.items():
            ax.plot(*zip(*pp.points), pp.fmt, label=label)
        ax.legend()
        self.fig.savefig(os.path.join(self.path, '{}.png'.format(name)))


class Statistics:
    # data_label -> PointsPlot
    _data: PointsPlots
    _file_prefix: str
    _path: str

    def __init__(self, path: str, file_prefix: str) -> None:
        self._path = path
        self._file_prefix = file_prefix
        self._data = defaultdict(lambda: PointsPlot([], ''))
        self._plot = Plot(path)

    def add_points(self, x: float, value_dict: typing.Dict[str, float]) -> None:
        for key, val in value_dict.items():
            self._data[key].points.append(Point(x, val))

    def plot(self, name: str, keys: typing.List[str] = None) -> None:
        fname = self._file_prefix+name
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

    def set_fmt(self, data_label: str, fmt: str) -> None:
        self._data[data_label].fmt = fmt

    def clear_graph(self):
        self._data.clear()


if __name__ == '__main__':
    s = Statistics('graphs', 'P')

    s.add_points(1, {'hey': 2, 'a': 3})
    s.add_points(2, {'hey': 4, 'a': 3})

    s.set_fmt('hey', '')
    s.set_fmt('a', 'ro')

    s.plot('O')
    s.plot('O_hey', ['hey'])
