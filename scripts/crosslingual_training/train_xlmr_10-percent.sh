#!/usr/bin/env bash

python3 srl/cross_lingual_train.py --name xlmr_10-percent_all --language_model xlm-roberta-base --config config/10_percent/all_config.json
python3 srl/cross_lingual_train.py --name xlmr_10-percent_ca --language_model xlm-roberta-base --config config/10_percent/ca_config.json
python3 srl/cross_lingual_train.py --name xlmr_10-percent_cz --language_model xlm-roberta-base --config config/10_percent/cz_config.json
python3 srl/cross_lingual_train.py --name xlmr_10-percent_de --language_model xlm-roberta-base --config config/10_percent/de_config.json
python3 srl/cross_lingual_train.py --name xlmr_10-percent_en --language_model xlm-roberta-base --config config/10_percent/en_config.json
python3 srl/cross_lingual_train.py --name xlmr_10-percent_es --language_model xlm-roberta-base --config config/10_percent/es_config.json
python3 srl/cross_lingual_train.py --name xlmr_10-percent_zh --language_model xlm-roberta-base --config config/10_percent/zh_config.json