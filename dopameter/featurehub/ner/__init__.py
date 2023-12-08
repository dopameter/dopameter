import collections
import logging


class NERFeatures:
    """Metrics by spaCy embedded NER metrics

    Parameters
    ----------
    nlp : spaCy nlp
    features : list

    Attributes
    ----------
    `data` : containing metrics of calculated NER metrics

    Define in configuration .json file under features:

      "tasks": ["features", "counts"],
      "features": {
        "ner": "default"
      }

    or in detail

      "tasks": ["features", "counts"],
      "features":
      {
        "ner": ["LOC", "ORG", "PERS", "MISC"],
      }

    """

    def __init__(
            self,
            nlp,
            features='default'
    ):

        logging.info('\tInitialize Named Entity Recognition (NER) features.')

        self.features = features

        if self.features != 'default':
            if set(self.features).intersection(nlp.get_pipe('ner').labels) == set():
                raise ValueError('Your NER features ' + ' '.join(self.features) + ' are not defined! Check the definitions: https://spacy.io/usage/linguistic-features')
            else:
                logging.info('\t\tDefined NER-tags: ' + str(set(self.features).intersection(nlp.get_pipe('ner').labels)))


    def feat_doc(self, doc):

        """Compute metrics of a Named Entity Relations (NER) of a document

        Parameters
        ----------
        doc : spaCy Doc

        Returns
        -------
        dict
            dictionary counts of NER occurrences
        """

        if self.features == 'default':
            ner = dict(collections.Counter([ent.label_ for ent in doc.ents]))
        else:
            ner = dict(collections.Counter([ent.label_ for ent in doc.ents if ent.label_ in self.features]))

        return {'counts': ner}
