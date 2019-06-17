# NPRG045

## Dependencies:

Install the following packages:

TODO add repositories and aspell install (packages on debian?)
* wget
* git
* shell
* python>=3.6 (tested on python3.7.2)
* pip
* aspell with english, german and french dictionary


Optinally, it is possible to run everything in a virtual environment.
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

For running the pipeline with sample data execute the following line. Note this does not need aspell and geneea as a dependency, because the data is already preprocessed.

```
make run_sample
```

For a full run:

```
make run
```

It will first preprocess and denormalize data and then process them. Repeated run will only rerun the second part. `make run` runs the experiment stored in `experiments.yaml`

Resulting data can be found in `graphs/current_timestamp`
