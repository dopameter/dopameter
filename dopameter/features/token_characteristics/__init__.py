import logging


class TokenCharacteristics:
    """Metrics by spaCy embedded linguistic features, here named 'token characteristics'
        * https://spacy.io/api/token
        * not used is_sent_start, is_sent_end

    Parameters
    ----------
    features : dict

    Attributes
    ----------
    `data` : containing metrics of calculated token characteristics metrics

    Define in configuration .json file under features:

      "tasks": ["features", "counts"],
      "features": {
        "token_characteristics": "default"
      }

    or in detail

      "tasks": ["features", "counts"],
      "features":
      {
        "token_characteristics":
            ["is_alpha", "is_ascii", "is_digit", "is_lower", "is_upper", "is_title", "is_punct",
             "is_left_punct", "is_right_punct", "is_space", "is_bracket", "is_quote", "is_currency",
             "like_url", "like_num", "like_email", "is_oov", "is_stop"]
      }

    """

    def __init__(
            self,
            features='default'
    ):

        logging.info('\tInitialize token character features.')

        default_features = ['is_alpha', 'is_ascii', 'is_digit', 'is_lower', 'is_upper', 'is_title', 'is_punct',
                             'is_left_punct', 'is_right_punct', 'is_space', 'is_bracket', 'is_quote', 'is_currency',
                             'like_url', 'like_num', 'like_email', 'is_oov', 'is_stop']
        if features == 'default':
            self.features = default_features
        else:
            if set(self.features).intersection(default_features) == set():
                raise ValueError('Your token character features ' + ' '.join(self.features) + ' are not defined! Allowed definitions: ', default_features)
            else:
                logging.info('\t\tDefined features: ' + features)
                self.features = features


    def feat_doc(self, doc):

        """Compute metrics by spaCy embedded linguistic features ('token characteristics') of a document

        Parameters
        ----------
        doc : spaCy Doc

        Returns
        -------
        dict
            dictionary counts of token characteristics occurrences
        """

        data = {'counts': {}}

        if 'is_alpha' in self.features:
            data['counts']['is_alpha'] = len([tok for tok in doc if tok.is_alpha])

        if 'is_ascii' in self.features:
            data['counts']['is_ascii'] = len([tok for tok in doc if tok.is_ascii])

        if 'is_digit' in self.features:
            data['counts']['is_digit'] = len([tok for tok in doc if tok.is_digit])

        if 'is_lower' in self.features:
            data['counts']['is_lower'] = len([tok for tok in doc if tok.is_lower])

        if 'is_upper' in self.features:
            data['counts']['is_upper'] = len([tok for tok in doc if tok.is_upper])

        if 'is_title' in self.features:
            data['counts']['is_title'] = len([tok for tok in doc if tok.is_title])

        if 'is_punct' in self.features:
            data['counts']['is_punct'] = len([tok for tok in doc if tok.is_punct])

        if 'is_left_punct' in self.features:
            data['counts']['is_left_punct'] = len([tok for tok in doc if tok.is_left_punct])

        if 'is_right_punct' in self.features:
            data['counts']['is_right_punct'] = len([tok for tok in doc if tok.is_right_punct])

        if 'is_space' in self.features:
            data['counts']['is_space'] = len([tok for tok in doc if tok.is_space])

        if 'is_bracket' in self.features:
            data['counts']['is_bracket'] = len([tok for tok in doc if tok.is_bracket])

        if 'is_quote' in self.features:
            data['counts']['is_quote'] = len([tok for tok in doc if tok.is_quote])

        if 'is_currency' in self.features:
            data['counts']['is_currency'] = len([tok for tok in doc if tok.is_currency])

        if 'like_url' in self.features:
            data['counts']['like_url'] = len([tok for tok in doc if tok.like_url])

        if 'like_num' in self.features:
            data['counts']['like_num'] = len([tok for tok in doc if tok.like_num])

        if 'like_email' in self.features:
            data['counts']['like_email'] = len([tok for tok in doc if tok.like_email])

        if 'is_oov' in self.features:
            data['counts']['is_oov'] = len([tok for tok in doc if tok.is_oov])

        if 'is_stop' in self.features:
            data['counts']['is_stop'] = len([tok for tok in doc if tok.is_stop])

        return data
