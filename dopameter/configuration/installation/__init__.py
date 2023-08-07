import requests
import json
import os
import zipfile
import logging


class ConfLanguages:

    """Install Language dependent features:

    Notes
    -----
    * spaCy models:
        * https://spacy.io/usage/models
    * Open Wordnet:
        * https://pypi.org/project/wn/
        * https://github.com/goodmami/wn
    * Constituency Parser / Berkeley Parser:
        * https://github.com/nikitakit/self-attentive-parser#available-models
    * MEmoLon Resources (emotion):
        * https://zenodo.org/record/6025889/#.YiW6MhsxnJk
    """

    def __init__(self):

        self.config_lang = json.load(

            open(
                os.path.join(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-4]),
                'ext_res',
                'installation',
                'config_lang.json')
                )
        )

        self.spacy_languages = self.config_lang['spacy_languages']
        self.benepar_languages = self.config_lang['benepar_languages']
        self.wordnet_languages = self.config_lang['wordnet_languages']
        self.lang_def = self.config_lang['spacy_languages_def']
        self.heideltime_languages = self.config_lang['heideltime_languages']

    def install_language_model(self, lang):
        logging.info('\t language model')
        if lang in self.spacy_languages.keys():
            os.system('python -m spacy download ' + self.spacy_languages[lang])
            return 0
        else:
            logging.info(lang + ' is not available as spaCy language module.')
            return -1

    def install_semantic_relations_wn_language_model(self, lang):
        logging.info('\t wordnet model for semantic relations')
        if lang in self.wordnet_languages.keys():
            import wn
            wn.download(self.wordnet_languages[lang])
            return 0
        else:
            logging.info(lang + ' is not available in the Open Wordnet (wn) modules.')
            return -1

    def install_constituency_language_model(self, lang):
        logging.info('\t language model for constituency')
        if lang in self.benepar_languages.keys():
            import benepar
            benepar.download(self.benepar_languages[lang])
            return 0
        else:
            logging.info(lang + ' is not available in the Berkeley Parser (benepar) modules.')
            return -1

    def install_emotion_lexicons(self, lang):
        logging.info('\t emotion lexicons')

        emotion_path = os.path.join(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-4]), 'ext_res', 'emotion')
        if not os.path.exists(emotion_path):
            os.makedirs(emotion_path)

        emotion_lang_path = os.path.join(emotion_path, lang + '.tsv.zip')

        em_target_path = emotion_lang_path.replace(os.sep + lang + '.tsv.zip', '')
        em_target_file = emotion_lang_path.replace(os.sep + lang + '.tsv.zip', '') + os.sep + lang + '.tsv'

        if not os.path.exists(em_target_file):
            emotion_file = 'https://zenodo.org/record/6025889/files/' + lang + '.tsv.zip'
            emotion_pack = requests.get(emotion_file, allow_redirects=True)
            if emotion_pack.status_code == 200:
                open(emotion_lang_path, 'wb').write(emotion_pack.content)
                with zipfile.ZipFile(emotion_lang_path, "r") as zip_ref:
                    zip_ref.extractall(em_target_path)
                os.remove(emotion_lang_path)
                logging.info(em_target_file + ' downloaded successfully.')
                return 0
            else:
                logging.info("Download of MEmoLon pack '" + lang + "' with a .tsv file is not possible.")
                logging.info("Check the path " + emotion_file + " and your defined language.")
                return -1
        else:
            logging.info(em_target_file, 'already exists.')
            return 0

    def install_all_possible_language_modules(self, lang):
        self.install_language_model(lang)
        self.install_semantic_relations_wn_language_model(lang)
        self.install_constituency_language_model(lang)
        self.install_emotion_lexicons(lang)
