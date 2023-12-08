import glob
import logging
import os

from dopameter.configuration.installation import ConfLanguages
from dopameter.language_collection import LanguageCollection
from dopameter.language_collection.corpora_collection import CorporaCollection
from dopameter.language_collection.corpora_collection.corpora.corpus import Corpus

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

        collections_corpora = {}

        for corpus_name in self.corpora:

            lang = self.corpora[corpus_name]['language']

            if ConfLanguages().lang_def[lang]:

                if lang not in collections_corpora:
                    collections_corpora[lang] = LanguageCollection(lang=lang, features=self.features)

                if 'file_formats' in self.corpora[corpus_name].keys():
                    corpus_files = []
                    for file_format in self.corpora[corpus_name]['file_formats']:
                        if file_format:
                            corpus_files += glob.glob(self.corpora[corpus_name]['path_text_data'] + os.sep + '*.' + file_format)
                        else:
                            corpus_files += glob.glob(self.corpora[corpus_name]['path_text_data'] + os.sep + '*')
                else:
                    corpus_files = glob.glob(self.corpora[corpus_name]['path_text_data'] + os.sep + '*.txt')

                if 'encoding' in self.corpora[corpus_name].keys():
                    encoding = self.corpora[corpus_name]['encoding']
                else:
                    encoding = 'utf-8'

                if 'collection' in self.corpora[corpus_name].keys():
                    collection_name = self.corpora[corpus_name]['collection']
                else:
                    collection_name = 'None'

                if collection_name not in collections_corpora[lang].collections:
                    collections_corpora[lang].collections[collection_name] = CorporaCollection(
                        lang=lang,
                        collection_name=collection_name,
                        features=self.features
                    )

                if corpus_files:
                    collections_corpora[lang].collections[collection_name].corpora.append(
                        Corpus(
                            corpus_path = self.corpora[corpus_name]['path_text_data'],
                            lang = lang,
                            name = corpus_name,
                            files = sorted(corpus_files),
                            encoding = encoding,
                            features = self.features,
                            external_resources = self.external_resources,
                            collection_name = collection_name
                        )
                    )

                else:
                    logging.info("The corpus '" + corpus_name + "' has no files. It will be not processed.")
            else:
                logging.info("The language of the corpus '" + corpus_name + "' is not a valid language for usage with spaCy. It will be not processed.")

        return collections_corpora
