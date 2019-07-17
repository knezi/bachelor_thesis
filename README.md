# bachelor thesis

TOPIC: Comparison of Approaches to Text Classification

The project was developed in Arch Linux and should work in any standard distribution,
so long all dependencies are installed.

## Dependencies:

Install the following software (for most distributions the name of a package in repositories should be the same):

* wget
* git
* shell
* python>=3.7 (tested on python3.7.2)
* pip
* aspell with english, german and french dictionary

Pip is usually a package python-pip or python3-pip.
It is important to install the pip version for python3.
It is also important to have python of a version at least 3.7.
Version 3.6 or bellow (python3 in Debian Stretch) will not work.

Aspell and dictionaries are usually packages `aspell` and `aspell-{en,fr,de}`.
https://github.com/pythonhttps://github.com/python
Optionally, it is possible to run the project in a virtual environment.
python3 can be replaced with the intended version of the python interpreter. Must be at least 3.7.

```
virtualenv venv --python=python3
cd venv
source ./bin/activate
```

Clone the repository and change working directory to the root:

```
git clone https://github.com/knezi/NPRG045
cd NPRG045
```

Install python dependencies from pip:

```
pip install `cat pip_deps`
```

## Execution:

Everything can be operated by make. Make expects the dataset from Yelp to be extracted in `../data/dataset`.

Next create the directory graphs:

```
mkdir graphs
```

For running the pipeline with sample data execute the following line. Note this does not need aspell as a dependency, because the data is already preprocessed.

```
make run_sample
```

For a full run:

```
make run
```

It will first preprocess and denormalize data and then process them. Repeated run will only rerun the second part. `make run` runs all experiments in directory `experiments`.


Resulting data can be found in `graphs/current_timestamp`
