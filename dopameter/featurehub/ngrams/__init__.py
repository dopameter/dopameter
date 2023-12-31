import collections
import textacy
import logging


class NGramFeatures:
    """Metrics calculated by ngrams - group of n words

    Parameters
    ----------
    features : list

    Attributes
    ----------
    `data`: containing counts of ngrams and document wise ngrams

    """

    def __init__(
            self,
            config,
            features=['default']
    ):

        logging.info('\tInitialize ngram features.')

        if features != 'default':
            if not all(isinstance(num, int) for num in features):
                raise ValueError('Your n-gram features ' + ' '.join(self.features) + ' are wrong! Allowed example [1,2,3]')
            self.features = features
        else:
            self.features = [1, 2, 3]
        self.config = config


    def feat_doc(self, doc):

        """Compute metrics of a POS (Part of Speech) of a document

        Parameters
        ----------
        doc : spaCy Doc

        Returns
        -------
        dict
            dictionary counts of NER occurrences and ngrams of the documents
        """

        data = {'ngrams': {}, 'counts': {}}

        for n in self.features:

            if not 'mode_ngrams' in self.config['settings'].keys():

                data['ngrams'][n] = dict(collections.Counter([
                    str(ngram) for ngram in list(textacy.extract.ngrams(
                        doc,
                        n=int(n),
                        filter_stops=False,
                        filter_punct=False,
                        filter_nums=False))
                    ]))
            else:
                if self.config['settings']['mode_ngrams'] == 'pos':
                    data['ngrams'][n] = dict(collections.Counter([
                        str(ngram) for ngram in doc._.pos_tags
                        ]))
                else:
                    ValueError('The configuration of your ngrams is wrong.')

            data['counts'][str(n) + '_grams'] = len(data['ngrams'][n])

        return data
