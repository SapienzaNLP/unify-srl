#!/bin/bash

python3 srl/train.py --name bert-multilingual-en
python3 srl/train.py --name bert-base-en --language_model bert-base-cased
python3 srl/train.py --name bert-large-en --language_model bert-large-cased

python3 srl/train.py --name bert-multilingual-es --train_path data/json/es/CoNLL2009_train.json --dev_path data/json/es/CoNLL2009_dev.json
python3 srl/train.py --name bert-multilingual-ca --train_path data/json/ca/CoNLL2009_train.json --dev_path data/json/ca/CoNLL2009_dev.json
python3 srl/train.py --name bert-multilingual-zh --train_path data/json/zh/CoNLL2009_train.json --dev_path data/json/zh/CoNLL2009_dev.json
python3 srl/train.py --name bert-multilingual-de --train_path data/json/de/CoNLL2009_train.json --dev_path data/json/de/CoNLL2009_dev.json
python3 srl/train.py --name bert-multilingual-cz --train_path data/json/cz/CoNLL2009_train.json --dev_path data/json/cz/CoNLL2009_dev.json --batch_size 16 --accumulate_grad_batches 2




