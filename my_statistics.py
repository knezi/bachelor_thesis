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

from statistics import mean, stdev
from typing import List, Dict, Generator, Set, ItemsView

from matplotlib.axes import Axes
from recordclass import RecordClass

from matplotlib import pyplot
import os


class Point:
    """Represent a point in a graph."""

    def __init__(self) -> None:
        self._values = []

    def add_value(self, val: float) -> None:
        self._values.append(val)

    def property(self, prp: str) -> float:
        """Return statistical property of contained data.

        It can return:
        min
        max
        mean
        stdev


        :param prp: the name of the property
        :returns: the value of the property
        """
        if prp == 'min':
            return min(self._values)

        if prp == 'max':
            return max(self._values)

        if prp == 'mean':
            return mean(self._values)

        if prp == 'stdev':
            return stdev(self._values)


class DataLine(RecordClass):
    """Store data points and format specs for a one type of data.

    points - dictionary of x coordinate -> RecordClasses Point
    fmt - format string for data plotting in mathplotilb"""
    points: Dict[float, Point]
    fmt: str = ''


DataLines = Dict[str, DataLine]
"""Typename for a dictionary holding several DataLine_s accesible by names.
Used for plotting graphs with more datalines."""  # TODO how to do this?


class DataGraph:
    """Class for buffering data points and other graph related properties.

        DataGraph - accept data as it goes from the programme flow, aggregate it
        and can be passed to Statistics as a whole for plotting"""
    _keys: Dict[str, Set]
    name: str
    # DataLines are stored in dict, each element represent a namespace
    _data: Dict[str, DataLines]

    def __init__(self, name: str = '', xlabel: str = '', ylabel: str = '') -> None:
        """Init empty data object.

        :param name: name.{png,csv} will be used for graph&csv dump respectively
        :param xlabel: label of x axis
        :param ylabel: label of y axis
        """
        self._data \
            = defaultdict(lambda:
                          defaultdict(lambda:
                                      DataLine(defaultdict(Point), '')))
        self.name = name
        self.xlabel = xlabel
        self.ylabel = ylabel
        self._keys = None

    def add_points(self, x: float, namespace: str, value_dict: Dict[str, float]) \
            -> None:
        """Add points with the same x and various y for different types.

        Y values None are replaced with dummy -1

        :param namespace: all points are accessed by namespace.name
        :param x: x value same for all points
        :param value_dict: dictionary 'the name of the datatype' -> 'y value'
        """
        for key, val in value_dict.items():
            if val is None:
                val = -1
            self._data[namespace][key].points[x].add_value(val)

    def set_fmt(self, namespace: str, data_type: str, fmt: str) -> None:
        """Set fmt string for given type (default empty).

        :param namespace: namespace of the type
        :param data_type: type of data
        :param fmt: formatting directive for mathplotlib
        """
        self._data[namespace][data_type].fmt = fmt

    def clear_data(self):
        """Empty DataLines container."""
        self._data.clear()

    def set_view(self, keys: Dict[str, Set]) -> None:
        """Set keys visible when dumping data with get_data.

        :param keys: dictionary of namespaces, values are individual keys"""
        self._keys = keys

    def get_data(self) -> Dict[str, DataLines]:
        """Returns DataLines of data from restricted view.

        The view is set by self.restrict_view"""

        if self._keys is not None:
            res = dict()
            for ns in self._keys:
                res[ns] = {key: self._data[ns][key] for key in self._keys[ns]}
            return res
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
        fname: str = self._file_prefix + data.name

        # Each graph has exactly one axes for plotting
        self._fig.clf()
        ax: Axes = self._fig.subplots()

        # setting Axes
        ax.set_xlabel(data.xlabel)
        ax.set_ylabel(data.ylabel)
        pps: Dict[str, DataLines] = data.get_data()
        for ns in pps:
            for label, pp in pps[ns].items():
                # convert data from [(x1,y1),...] to [[x1,x2...], [y1, y2...]]
                means = map(lambda a: (a[0], a[1].property('mean')), pp.points.items())
                ax.plot(*zip(*means), pp.fmt, label=f'{ns}.{label}')
        ax.legend()

        self._fig.savefig(os.path.join(self._path, f'{fname}.png'))

        # dump textual representation in CSV, columns being:
        props: tuple = ('mean', 'min', 'max', 'stdev')
        header_y: str = ';'.join(props)
        # property1,x;prop1.y;...;propn.x;propn.y
        # every y consists of mean;min;max;stdev as defined in props
        # convert keys to tuples to preserve the order
        # it's ((namespace, (keys)), (namespace2, (keys2))...)
        keys_ordered = tuple()
        for ns in pps:
            keys_ordered += ((ns, (key for key in pps[ns])),)

        # dump of points in format p.x;p.y
        # points are grouped in lines on x axes
        with open(os.path.join(self._path, f'{fname}.csv'), 'w') as w:
            # header
            # y is left for Y axis
            # it produces:
            # namespace1.key1;y;namespace1.key2;y;....nsn.key1;y;nsnkey2...
            for (ns, ks) in keys_ordered:
                w.write(f';{header_y};'.join((f'{ns}.{k}' for k in ks)))
                w.write(f';{header_y};')
            w.write('\n')

            # convert data into iterable of lines being list of strings
            # and round values to three digits
            # zip(*iterable) reverses columns and rows
            def convert_points_to_list(points: ItemsView[float, Point]) -> List[str]:
                """Convert list of x,point to list of 'x;mean;min;max;...'

                Take [1, Point()] and return ['1;point.mean();point.min()...']
                """
                res = []
                for x, y in points:
                    y_vals: str \
                        = ';'.join([str(round(y.property(prp), 3)) for prp in props])
                    p_str: str = f'{round(x, 3)};{y_vals}'
                    res.append(p_str)
                return res

            data_lists: Generator[List[str], None, None] \
                = zip(*map(convert_points_to_list,
                           [pps[ns][k].points.items()
                            for (ns, keys) in pps.items() for k in keys]))

            # convert data into lines (str)
            data_lines: Generator[str, None, None] = map(';'.join, data_lists)

            for line in data_lines:
                w.write(line + '\n')


# only for testing purposes
if __name__ == '__main__':
    s = Statistics('graphs', 'P')

    dg = DataGraph()
    dg.add_points(1, 'n', {'hey': 2, 'a': 3})
    dg.add_points(2, 'n', {'hey': 4, 'a': 3})
    dg.add_points(1, 'n', {'hey': 2, 'a': 3})
    dg.add_points(2, 'n', {'hey': 3, 'a': 3})

    dg.set_fmt('n', 'hey', '')
    dg.set_fmt('n', 'a', 'ro')

    dg.name = 'trial'
    s.plot(dg)
    dg.name = 'trial2'
    dg.set_view({'n': {'hey'}})
    s.plot(dg)

    p = Point()
    p.add_value(1)
    assert (p.property('min') == 1)
    assert (p.property('max') == 1)
    assert (p.property('mean') == 1)
    p.add_value(2)
    assert (p.property('min') == 1)
    assert (p.property('max') == 2)
    assert (p.property('mean') == 1.5)
    assert (abs(p.property('stdev') ** 2 - 0.5) < 0.0001)
