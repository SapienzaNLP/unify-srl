#!/bin/bash

python3 srl/train.py --name xlmr-en --language_model xlm-roberta-base
python3 srl/train.py --name xlmr-es --language_model xlm-roberta-base --train_path data/json/es/CoNLL2009_train.json --dev_path data/json/es/CoNLL2009_dev.json
python3 srl/train.py --name xlmr-ca --language_model xlm-roberta-base --train_path data/json/ca/CoNLL2009_train.json --dev_path data/json/ca/CoNLL2009_dev.json
python3 srl/train.py --name xlmr-zh --language_model xlm-roberta-base --train_path data/json/zh/CoNLL2009_train.json --dev_path data/json/zh/CoNLL2009_dev.json --batch_size 16 --accumulate_grad_batches 2
python3 srl/train.py --name xlmr-de --language_model xlm-roberta-base --train_path data/json/de/CoNLL2009_train.json --dev_path data/json/de/CoNLL2009_dev.json
python3 srl/train.py --name xlmr-cz --language_model xlm-roberta-base --train_path data/json/cz/CoNLL2009_train.json --dev_path data/json/cz/CoNLL2009_dev.json
