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
Version 3.5 (python3 in Debian Stretch) will not work.

Python can be manually installed by:

apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

```
wget https://www.python.org/ftp/python/3.6.3/Python-3.6.3.tgz
tar xvf Python-3.6.3.tgz
cd Python-3.6.3
./configure --enable-optimizations
make -j8
sudo make altinstall
```

Aspell and dictionaries are usually packages `aspell` and `aspell-{en,fr,de}`.
https://github.com/pythonhttps://github.com/python
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
