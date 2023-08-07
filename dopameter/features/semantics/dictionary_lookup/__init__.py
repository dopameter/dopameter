import glob
import os
import logging
from spacy.matcher import PhraseMatcher


class DictionaryLookUp:
    """Metrics by Dictionary Lookup

    Parameters
    ----------
    nlp : spaCy nlp
    path_dictionaries : string
    file_format_dicts : string
    features : dict

    Attributes
    ----------
    `data` : containing metrics of calculated dictionary lookups

    Define in configuration .json file under features:

      "features": {
        "dictionary_lookup": ["examples_1", "examples_2"]
      },
      "external_resources":
      {
        "dictionaries":
        {
          "examples_1": "/path/to/dicts/examples_1",
          "examples_2": "/path/to/dicts/examples_2"
        }
      }

    """


    def __init__(
            self,
            nlp,
            path_dictionaries,
            file_format_dicts,
            features
    ):
        self.nlp = nlp
        self.features = features

        self.dictionary_paths = {}

        logging.info('\tInitialize dictionary based features.')
        logging.info('\tDictionary file format: ' + file_format_dicts)

        if features == 'default':
            self.features = list(path_dictionaries.keys())
        else:
            if set(features).intersection(set(path_dictionaries.keys())):
                self.features == features
            else:
                raise ValueError('Your configuration with dictionaries is wrong! Check your configuration file.')

        for dictionary in {path: path_dictionaries[path] for path in path_dictionaries if path in features}:
            self.dictionary_paths[dictionary] = {}
            for file in sorted(glob.glob(path_dictionaries[dictionary] + os.sep + '**/*.' + file_format_dicts, recursive=True)):
                self.dictionary_paths[dictionary][os.path.splitext(os.path.basename(file))[0]] = file

        self.phrase_matcher = {}
        for feat in self.dictionary_paths:
            self.phrase_matcher[feat] = PhraseMatcher(self.nlp.vocab, attr='LEMMA')
            for d in self.dictionary_paths[feat]:
                logging.info("\tdictionary '" + d + "' from " + self.dictionary_paths[feat][d])
                dictionary = [item.rstrip() for item in open(self.dictionary_paths[feat][d], 'r', encoding='utf8').readlines()]
                patterns = [nlp(entry) for entry in dictionary]
                self.phrase_matcher[feat].add(d, None, *patterns)
                logging.info("\t\tloaded with " + str(len(patterns)) + " patterns")

    def feat_doc(self, doc):

        """Compute metrics of a document with dictionary lookups

        Parameters
        ----------
        doc : spaCy Doc

        Returns
        -------
        dict
            dictionary with dictionary lookups, named by the features from the configuration
        """

        data = {}

        for feat in self.features:

            look_ups = self.phrase_matcher[feat](doc)
            data['dictionary_lookup_' + feat] = {}

            for dictionary in self.dictionary_paths[feat].keys():
                data['dictionary_lookup_' + feat][dictionary] = len([(doc[start:end]) for mat_id, start, end in look_ups if self.nlp.vocab.strings[mat_id] == dictionary])
        return data
