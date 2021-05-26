#!/usr/bin/env bash

python3 srl/cross_lingual_train.py --name bert_full_all --language_model bert-base-multilingual-cased --config config/full/all_config.json --language_model_fine_tuning --batch_size 16
python3 srl/cross_lingual_train.py --name bert_full_ca --language_model bert-base-multilingual-cased --config config/full/ca_config.json --language_model_fine_tuning --batch_size 16
python3 srl/cross_lingual_train.py --name bert_full_cz --language_model bert-base-multilingual-cased --config config/full/cz_config.json --language_model_fine_tuning --batch_size 8
python3 srl/cross_lingual_train.py --name bert_full_de --language_model bert-base-multilingual-cased --config config/full/de_config.json --language_model_fine_tuning --batch_size 16
python3 srl/cross_lingual_train.py --name bert_full_en --language_model bert-base-multilingual-cased --config config/full/en_config.json --language_model_fine_tuning --batch_size 16
python3 srl/cross_lingual_train.py --name bert_full_es --language_model bert-base-multilingual-cased --config config/full/es_config.json --language_model_fine_tuning --batch_size 16
python3 srl/cross_lingual_train.py --name bert_full_zh --language_model bert-base-multilingual-cased --config config/full/zh_config.json --language_model_fine_tuning --batch_size 16