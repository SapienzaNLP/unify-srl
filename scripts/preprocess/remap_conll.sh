#!/bin/bash

python3 scripts/preprocess/remap_conll.py --input data/json/en/CoNLL2009_train.json --conll_2009 --output data/json/en/CoNLL2009_train.va.json --mapping data/verbatlas/pb2va.tsv --frame_info data/verbatlas/VA_frame_info.tsv --log DEBUG
python3 scripts/preprocess/remap_conll.py --input data/json/en/CoNLL2009_dev.json --conll_2009 --output data/json/en/CoNLL2009_dev.va.json --mapping data/verbatlas/pb2va.tsv --frame_info data/verbatlas/VA_frame_info.tsv --log DEBUG
python3 scripts/preprocess/remap_conll.py --input data/json/en/CoNLL2009_test.json --conll_2009 --output data/json/en/CoNLL2009_test.va.json --mapping data/verbatlas/pb2va.tsv --frame_info data/verbatlas/VA_frame_info.tsv --log DEBUG