#!/bin/bash

python3 scripts/preprocess/preprocess_conll_2012.py --input data/txt/en/CoNLL2012_train.txt --output data/json/en/CoNLL2012_train.json
python3 scripts/preprocess/preprocess_conll_2012.py --input data/txt/en/CoNLL2012_dev.txt --output data/json/en/CoNLL2012_dev.json
python3 scripts/preprocess/preprocess_conll_2012.py --input data/txt/en/CoNLL2012_test.txt --output data/json/en/CoNLL2012_test.json