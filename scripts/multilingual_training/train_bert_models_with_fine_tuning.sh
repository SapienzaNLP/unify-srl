#!/bin/bash

python3 srl/train.py --name bert-multilingual-en --language_model_fine_tuning --language_model_learning_rate 1e-5 --language_model_min_learning_rate 1e-9 --batch_size 8 --accumulate_grad_batches 16
python3 srl/train.py --name bert-base-en-ft --language_model bert-base-cased --language_model_fine_tuning --language_model_learning_rate 1e-5 --language_model_min_learning_rate 1e-9 --batch_size 16 --accumulate_grad_batches 8
python3 srl/train.py --name bert-large-en-ft --language_model bert-large-cased --language_model_fine_tuning --language_model_learning_rate 1e-5 --language_model_min_learning_rate 1e-9 --batch_size 8 --accumulate_grad_batches 16

python3 srl/train.py --name bert-multilingual-es-ft --train_path data/json/es/CoNLL2009_train.json --dev_path data/json/es/CoNLL2009_dev.json --language_model_fine_tuning --language_model_learning_rate 1e-5 --language_model_min_learning_rate 1e-9 --batch_size 8 --accumulate_grad_batches 16
python3 srl/train.py --name bert-multilingual-ca-ft --train_path data/json/ca/CoNLL2009_train.json --dev_path data/json/ca/CoNLL2009_dev.json --language_model_fine_tuning --language_model_learning_rate 1e-5 --language_model_min_learning_rate 1e-9 --batch_size 8 --accumulate_grad_batches 16
python3 srl/train.py --name bert-multilingual-zh-ft --train_path data/json/zh/CoNLL2009_train.json --dev_path data/json/zh/CoNLL2009_dev.json --language_model_fine_tuning --language_model_learning_rate 1e-5 --language_model_min_learning_rate 1e-9 --batch_size 8 --accumulate_grad_batches 16
python3 srl/train.py --name bert-multilingual-de-ft --train_path data/json/de/CoNLL2009_train.json --dev_path data/json/de/CoNLL2009_dev.json --language_model_fine_tuning --language_model_learning_rate 1e-5 --language_model_min_learning_rate 1e-9 --batch_size 8 --accumulate_grad_batches 16
python3 srl/train.py --name bert-multilingual-cz-ft --train_path data/json/cz/CoNLL2009_train.json --dev_path data/json/cz/CoNLL2009_dev.json --language_model_fine_tuning --language_model_learning_rate 1e-5 --language_model_min_learning_rate 1e-9 --batch_size 8 --accumulate_grad_batches 16
