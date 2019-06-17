# NPRG045

The project was developed in Linux.
The author uses Arch Linux.
It was also tested on a standard debian installation (debian Stretch).
It may work on other platform, but beware that it has not been tested.

## Dependencies:

Install the following software (for most distributions the name of a package in repositories should be the same):

* wget
* git
* shell
* python>=3.6 (tested on python3.7.2)
* pip
* aspell with english, german and french dictionary

Pip is usually a package python-pip or python3-pip.
It is important to install the pip version for python3.
It is also important to have python of a version at least 3.6.
Version 3.5 (python3 in Debian Stretch - 9) will not work.

Aspell with dictionaries is usally is packages `aspell`, `aspell-{en,fr,de}`.

Optionally, it is possible to run the project in a virtual environment.
python3 can be replaced with the installed version. Must be at least 3.6.

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

Everything can be operated by make.

For running the pipeline with sample data execute the following line. Note this does not need aspell as a dependency, because the data is already preprocessed.

```
make run_sample
```

For a full run:

```
make run
```

It will first preprocess and denormalize data and then process them. Repeated run will only rerun the second part. `make run` runs the experiment stored in `experiments.yaml`

Resulting data can be found in `graphs/current_timestamp`
