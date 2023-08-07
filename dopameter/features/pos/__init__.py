import collections
import logging


class POSFeatures:

    """Metrics by spaCy embedded POS metrics

    Parameters
    ----------
    nlp : spaCy nlp
    features : dict

    Attributes
    ----------
    `data` : containing metrics of calculated POS metrics

    Define in configuration .json file under features:

      "tasks": ["features", "counts"],
      "features": {
        "pos": "default"
      }

    or in detail

      "tasks": ["features", "counts"],
      "features":
      {
        "pos": ["NNP", "VBG", "NN", "$"]
      }

    """

    def __init__(
            self,
            nlp,
            features='default'
    ):

        """
            https://spacy.io/usage/linguistic-features
        """

        logging.info('\tInitialize Part-of-Speech tagging (POS) features.')

        self.features = features

        if self.features != 'default':
            if set(self.features).intersection(nlp.get_pipe('pos').labels) == set():
                raise ValueError('Your POS features ' + ' '.join(self.features) + ' are not defined! Check the definitions: https://spacy.io/usage/linguistic-features')
            else:
                logging.info('\t\tDefined POS-tags: ' + str(set(self.features).intersection(nlp.get_pipe('pos').labels)))


    def feat_doc(self, doc):

        """Compute metrics of a POS (Part of Speech) of a document

        Parameters
        ----------
        doc : spaCy Doc

        Returns
        -------
        dict
            dictionary counts of NER occurrences
        """

        if self.features == 'default':
            pos = dict(collections.Counter([str(token.pos_) for token in doc]))
        else:
            pos = dict(collections.Counter([str(token.pos_) for token in doc if str(token.pos_) in self.features]))

        return {'counts': pos}
