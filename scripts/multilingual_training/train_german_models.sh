#!/bin/bash

python3 srl/train.py --name bert-multilingual-de-90 --train_path data/json/de/CoNLL2009_train.json --dev_path data/json/de/CoNLL2009_dev.json --limit_train_batches 0.9
python3 srl/train.py --name bert-multilingual-de-80 --train_path data/json/de/CoNLL2009_train.json --dev_path data/json/de/CoNLL2009_dev.json --limit_train_batches 0.8
python3 srl/train.py --name bert-multilingual-de-70 --train_path data/json/de/CoNLL2009_train.json --dev_path data/json/de/CoNLL2009_dev.json --limit_train_batches 0.7
python3 srl/train.py --name bert-multilingual-de-60 --train_path data/json/de/CoNLL2009_train.json --dev_path data/json/de/CoNLL2009_dev.json --limit_train_batches 0.6
python3 srl/train.py --name bert-multilingual-de-50 --train_path data/json/de/CoNLL2009_train.json --dev_path data/json/de/CoNLL2009_dev.json --limit_train_batches 0.5
python3 srl/train.py --name bert-multilingual-de-25 --train_path data/json/de/CoNLL2009_train.json --dev_path data/json/de/CoNLL2009_dev.json --limit_train_batches 0.25
python3 srl/train.py --name bert-multilingual-de-10 --train_path data/json/de/CoNLL2009_train.json --dev_path data/json/de/CoNLL2009_dev.json --limit_train_batches 0.1
