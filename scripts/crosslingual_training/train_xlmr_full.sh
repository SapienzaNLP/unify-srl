#!/usr/bin/env bash

python3 srl/cross_lingual_train.py --name xlmr_full_all --language_model xlm-roberta-base --config config/full/all_config.json
python3 srl/cross_lingual_train.py --name xlmr_full_ca --language_model xlm-roberta-base --config config/full/ca_config.json
python3 srl/cross_lingual_train.py --name xlmr_full_cz --language_model xlm-roberta-base --config config/full/cz_config.json
python3 srl/cross_lingual_train.py --name xlmr_full_de --language_model xlm-roberta-base --config config/full/de_config.json
python3 srl/cross_lingual_train.py --name xlmr_full_en --language_model xlm-roberta-base --config config/full/en_config.json
python3 srl/cross_lingual_train.py --name xlmr_full_es --language_model xlm-roberta-base --config config/full/es_config.json
python3 srl/cross_lingual_train.py --name xlmr_full_zh --language_model xlm-roberta-base --config config/full/zh_config.json