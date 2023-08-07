import argparse
import logging
import os
from datetime import date
import json


from dopameter.configuration.installation import ConfLanguages

if __name__ == '__main__':

    if not os.path.isdir('log'):
        os.mkdir('log')

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler('log' + os.sep + 'logs_feattext_' + str(date.today()) + '.log'),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('json')
    args = parser.parse_args()

    if os.path.exists(args.json):

        with open(args.json, encoding='utf-8') as json_file:
            try:
                config = json.load(json_file)
                logging.info(args.json + ' loaded.')
            except:
                exit('Your configuration file ' + args.json + ' file is not a valid json-file. Routine aborted.')

    inst = ConfLanguages()
    for lang in config['languages']:
        logging.info('install ' + inst.lang_def[lang])
        if 'lang' in config['modules']:
            logging.info('Install spaCy language model')
            inst.install_semantic_relations_wn_language_model(lang)
        if 'wordnet' in config['modules']:
            logging.info('Install wordnet (wn)')
            inst.install_semantic_relations_wn_language_model(lang)
        if 'const' in config['modules']:
            logging.info('Install constituency parser benepar')
            inst.install_constituency_language_model(lang)
        if 'emotion' in config['modules']:
            logging.info('Install emotion lexicons')
            inst.install_emotion_lexicons(lang)