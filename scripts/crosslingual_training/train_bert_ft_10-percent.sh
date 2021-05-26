#!/usr/bin/env bash

python3 srl/cross_lingual_train.py --name bert_10-percent_all --language_model bert-base-multilingual-cased --config config/10_percent/all_config.json --language_model_fine_tuning --batch_size 16 --accumulate_grad_batches 8
python3 srl/cross_lingual_train.py --name bert_10-percent_ca --language_model bert-base-multilingual-cased --config config/10_percent/ca_config.json --language_model_fine_tuning --batch_size 16 --accumulate_grad_batches 8
python3 srl/cross_lingual_train.py --name bert_10-percent_cz --language_model bert-base-multilingual-cased --config config/10_percent/cz_config.json --language_model_fine_tuning --batch_size 8 --accumulate_grad_batches 16
python3 srl/cross_lingual_train.py --name bert_10-percent_de --language_model bert-base-multilingual-cased --config config/10_percent/de_config.json --language_model_fine_tuning --batch_size 16 --accumulate_grad_batches 8
python3 srl/cross_lingual_train.py --name bert_10-percent_en --language_model bert-base-multilingual-cased --config config/10_percent/en_config.json --language_model_fine_tuning --batch_size 16 --accumulate_grad_batches 8
python3 srl/cross_lingual_train.py --name bert_10-percent_es --language_model bert-base-multilingual-cased --config config/10_percent/es_config.json --language_model_fine_tuning --batch_size 16 --accumulate_grad_batches 8
python3 srl/cross_lingual_train.py --name bert_10-percent_zh --language_model bert-base-multilingual-cased --config config/10_percent/zh_config.json --language_model_fine_tuning --batch_size 16 --accumulate_grad_batches 8