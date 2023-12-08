import collections
import glob
import os
import logging


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

        self.dict_paths = {}

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
            self.dict_paths[dictionary] = {}
            for file in sorted(glob.glob(path_dictionaries[dictionary] + os.sep + '**/*.' + file_format_dicts, recursive=True)):
                self.dict_paths[dictionary][os.path.splitext(os.path.basename(file))[0]] = file

        self.dict_entries = {f : {} for f in self.features}

        for feat in self.dict_paths:
            for dict_file in self.dict_paths[feat]:
                logging.info("\tdictionary '" + dict_file + "' from " + self.dict_paths[feat][dict_file])
                dictionary = [item.rstrip() for item in open(self.dict_paths[feat][dict_file], 'r', encoding='utf8').readlines()]

                ext_dict = collections.defaultdict(list)
                for entry in dictionary:
                    dict_words = entry.split(' ')
                    ext_dict[dict_words[0]] += [dict_words]

                self.dict_entries[feat][dict_file] = ext_dict
                logging.info("\t\tloaded with " + str(len(dictionary)) + " entries.")

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
            matches = []
            entries = []
            entries_2 = collections.defaultdict(list)
            for dictionary in self.dict_entries[feat]:
                for i, token in enumerate(doc):
                    if token.lemma_ in self.dict_entries[feat][dictionary].keys():
                        for ent in self.dict_entries[feat][dictionary][token.lemma_]:
                            if ent == [t.lemma_ for t in doc[i:(i + len(ent))]]:
                                matches.append(dictionary)
                                entries.append(doc[i:(i + len(ent))].lemma_)
                                entries_2[dictionary].append(doc[i:(i + len(ent))].lemma_)

            entries_2 = {ent: collections.Counter(entries_2[ent]) for ent in entries_2}

            data['dictionary_lookup_' + feat] = {
                'counts': dict(collections.Counter(matches)),
                'entries': collections.Counter(entries),
                'entries_2': entries_2
            }

        return data
