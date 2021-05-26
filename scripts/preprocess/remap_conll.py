import argparse
import json
import logging
import os

from collections import Counter


DEFAULT_CONLL2009_ARGS_MAP = {
    "AM-EXT": "Extent",
    "AM-MNR": "Attribute",
    "AM-LOC": "Location",
    "AM-DIR": "Destination",
    "AM-REC": "Recursive",
    "AM-PNC": "Purpose",
    "AM-MOD": "Modal",
    "AM-TMP": "Time",
    "AM-PRT": "Predicative",
    "AM-NEG": "Negation",
    "AM-ADV": "Adverbial",
    "AM-CAU": "Cause",
    "AM-DIS": "Connective",
    "AM-PRD": "Predicative",
    "AM": "Modifier",
}

DEFAULT_CONLL2012_ARGS_MAP = {
    "ARGM-EXT": "Extent",
    "ARGM-MNR": "Attribute",
    "ARGM-LOC": "Location",
    "ARGM-DIR": "Destination",
    "ARGM-REC": "Recursive",
    "ARGM-PNC": "Purpose",
    "ARGM-MOD": "Modal",
    "ARGM-TMP": "Time",
    "ARGM-PRT": "Predicative",
    "ARGM-NEG": "Negation",
    "ARGM-ADV": "Adverbial",
    "ARGM-CAU": "Cause",
    "ARGM-DIS": "Connective",
    "ARGM-PRD": "Predicative",
    "ARGM": "Modifier",
}


def read_mapping(path, conll_2009=False):
    predicate_mapping = {}
    role_mapping = {}

    with open(path) as f:

        for line_index, line in enumerate(f):
            if line_index == 0:
                continue

            predicate, *roles = line.strip().split()
            pb_sense, va_sense = predicate.split('>')
            if pb_sense in predicate_mapping:
                continue
            if not conll_2009 and '2009' in pb_sense:
                continue
            elif conll_2009 and '2009' in pb_sense:
                pb_sense = pb_sense[:-5]

            predicate_mapping[pb_sense] = va_sense
            role_mapping[pb_sense] = {}

            for role in roles:
                pb_role, va_role = role.split('>')
                role_mapping[pb_sense][pb_role] = va_role
    
    return predicate_mapping, role_mapping


def read_frame_info(path):
    frame_names = {}

    with open(path) as f:

        for line_index, line in enumerate(f):
            if line_index == 0:
                continue

            frame_id, frame_name, *_ = line.strip().split('\t')
            frame_names[frame_id] = frame_name
    
    return frame_names


def remap(path, frame_names, predicate_mapping, role_mapping, use_default_mapping=False, include_RC_roles=False, conll_2009=False):
    with open(path) as f:
        data = json.load(f)
    
    verb_pos_set = {'VBD', 'VBZ', 'VBG', 'VBP', 'VBN', 'MD', 'VB'}
    num_errors = 0
    missing_predicates = []
    all_predicates = set()

    for _, sentence in data.items():
        predicates = sentence['predicates']
        pos_tags = sentence['pos_tags']
        roles = sentence['roles']

        va_predicates = []
        va_roles = {}

        for predicate_index, (predicate, pos_tag) in enumerate(zip(predicates, pos_tags)):
            if predicate == '_' or pos_tag not in verb_pos_set:
                va_predicates.append('_')
                continue

            if '-v' in predicate:
                predicate_lemma, predicate_number = predicate.split('.')
                predicate = '{}.{}'.format(predicate_lemma[:-2], predicate_number)

            if predicate not in predicate_mapping:
                va_predicates.append('_')
                num_errors += 1
                missing_predicates.append(predicate)
                logging.debug('Could not map PB sense ({}) to VA frame...'.format(predicate))
                continue

            va_frame = predicate_mapping[predicate]
            va_frame = frame_names[va_frame]
            va_predicates.append(va_frame)
            all_predicates.add(predicate)

            va_roles[predicate_index] = []
            for role in roles[str(predicate_index)]:
                bio_tag = ''
                if role != '_' and not conll_2009:
                    bio_tag = role[:2]
                    role = role[2:]
                
                role = role.replace('ARG', 'A')

                if include_RC_roles and (role[:2] == 'R-' or role[:2] == 'C-'):
                    role_type = role[:2]
                    role = role[2:]
                else:
                    role_type = None

                if role == 'V':
                    va_role = 'V'
                elif role in role_mapping[predicate]:
                    va_role = role_mapping[predicate][role]
                elif conll_2009 and use_default_mapping and role in DEFAULT_CONLL2009_ARGS_MAP:
                    va_role = DEFAULT_CONLL2009_ARGS_MAP[role]
                elif not conll_2009 and use_default_mapping and role in DEFAULT_CONLL2012_ARGS_MAP:
                    va_role = DEFAULT_CONLL2012_ARGS_MAP[role]
                elif role == '_':
                    va_role = '_'
                else:
                    va_role = '_'
                    if use_default_mapping or (use_default_mapping and 'AM' in role):
                        logging.debug('Could not map PB role ({} in {}) to VA...'.format(role, predicate))
                
                if include_RC_roles and role_type:
                    va_role = '{}{}'.format(role_type, va_role)
                if va_role != '_':
                    va_role = '{}{}'.format(bio_tag, va_role)
                
                va_roles[predicate_index].append(va_role)
            
        sentence['predicates'] = va_predicates
        sentence['roles'] = va_roles
    
    if num_errors > 0:
        missing_predicates = Counter(missing_predicates)
        logging.debug('Number of PB predicates that could not be remapped: {} (unique: {}/{})'.format(num_errors, len(missing_predicates), len(all_predicates)))
        for predicate, occurrences in missing_predicates.items():
            logging.debug('{}: {}'.format(predicate, occurrences))

    return data


def write_remapped_data(data, path):
    output = json.dumps(data, indent=4, sort_keys=True)

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'w') as f:
        f.write(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        dest='input_path',
        help='Path to the data to remap from PB to VA.')
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        dest='output_path',
        help='Path to the VA output file.')
    parser.add_argument(
        '--mapping',
        type=str,
        required=True,
        dest='mapping_path',
        help='Path to the mapping from PropBank to VerbAtlas.'
    )
    parser.add_argument(
        '--frame_info',
        type=str,
        required=True,
        dest='frame_info_path',
        help='Path to the frame info file (VA_frame_info.tsv).'
    )
    parser.add_argument(
        '--conll_2009',
        action='store_true',
        help='Whether the input is CoNLL-2009 (default: CoNLL-2012).'
    )
    parser.add_argument(
        '--log',
        type=str,
        default='WARNING',
        dest='loglevel',
        help='Log level. Default = WARNING.')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    logging.info('Parsing {}...'.format(args.input_path))

    frame_names = read_frame_info(args.frame_info_path)
    predicate_mapping, role_mapping = read_mapping(args.mapping_path, conll_2009=args.conll_2009)
    remapped_data = remap(args.input_path, frame_names, predicate_mapping, role_mapping, use_default_mapping=False, include_RC_roles=False, conll_2009=args.conll_2009)
    write_remapped_data(remapped_data, args.output_path)

    logging.info('Done!')