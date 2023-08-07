import glob
import logging
import os

from dopameter import ConfLanguages
from dopameter.corpora.corpus import Corpus


class SetUpCorpora:

    """Set up Corpora via single corpora

    Parameters
    ----------
    corpora : dict
    features : dict
    external_resources : dict

    Notes
    -----
    SetUpCorpora creates corpora by set up of single data collections of text files

    """

    def __init__(self, corpora, features, external_resources):
        self.corpora = corpora
        self.features = features
        self.external_resources = external_resources

    def create_corpora(self):
        """creates corpora by set up of single data collections of text files"""

        corpora = []
        for cor in self.corpora:

            lang = self.corpora[cor]['language']

            if ConfLanguages().lang_def[lang]:
                if 'file_formats' in self.corpora[cor].keys():
                    corpus_files = []
                    for file_format in self.corpora[cor]['file_formats']:
                        if file_format:
                            corpus_files += glob.glob(self.corpora[cor]['path_text_data'] + os.sep + '*.' + file_format)
                        else:
                            corpus_files += glob.glob(self.corpora[cor]['path_text_data'] + os.sep + '*')
                else:
                    corpus_files = glob.glob(self.corpora[cor]['path_text_data'] + os.sep + '*.txt')

                if 'encoding' in self.corpora[cor].keys():
                    encoding = self.corpora[cor]['encoding']
                else:
                    encoding = 'utf-8'

                if corpus_files:
                    corpora.append(
                        Corpus(
                            self.corpora[cor]['path_text_data'],
                            lang,
                            cor,
                            sorted(corpus_files),
                            encoding,
                            self.features,
                            self.external_resources
                        )
                    )
                else:
                    logging.info("The corpus '" + cor + "' has no files. It will be not processed.")
            else:
                logging.info("The language of the corpus '" + cor + "' is not a valid language for usage with spaCy. It will be not processed.")

        corpora.sort(key=lambda c: c.name)

        return corpora