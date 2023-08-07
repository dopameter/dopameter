import argparse
import json
import logging
import os
from datetime import date


from dopameter import DoPaMeter

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

        dpm = DoPaMeter(config=config)
        dpm.run_dopameter()

    else:
        exit('Your configuration file ' + args.json + ' is not existing.')
