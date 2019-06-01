RUN_DEP = fastText/fasttext ./exceptions.py ./preprocessors/__init__.py ./preprocessors/featurematrixconversion.py ./preprocessors/mutualinformation.py ./preprocessors/featureselectionbase.py ./preprocessors/dumpfasttext.py ./preprocessors/chisquare.py ./preprocessors/preprocessingbase.py ./statistics.py ./classifiers/__init__.py ./classifiers/baseline.py ./classifiers/fasttext.py ./classifiers/classifierbase.py ./classifiers/naivebayes.py ./utils.py ./load_data.py ./process_data.py

DATA_GEN_DEP = ./extract_ids.py ./filter.py ./reduce_bus.py ./get_data_for_ids.py ./compare_langs.py ./reduce_user.py ./join.py ./crop_geneea.py

data/data.json: $(DATA_GEN_DEP)
	# run denormalize

data/geneea.json: $(DATA_GEN_DEP)
	# run geenea

run: data/data.json data/geneea.json $(RUN_DEP)
	./process_data.py experiments.yaml data/data.json data/geneea.json
	
run_sample: data/data_sample.json data/geneea_sample.json $(RUN_DEP)
	./process_data.py experiments.yaml data/data_sample.json data/geneea_sample.json

clean:
	rm -f data/data_fasttext_model.{bin,vec} data/data_fasttext_train
	# add data/ids here once it's generated properly
	# todo data cleaning

test:
	./run_unittests.sh

all: run

.DEFAULT: all
.PHONY: all run run_sample clean
