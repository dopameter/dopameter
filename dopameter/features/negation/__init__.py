import glob
import os
import logging
from spacy.matcher import PhraseMatcher


class NegationFeatures:

    def __init__(
        self,
        nlp,
        lang,
        features = 'default'
        ):

        """
            this package is derived from
            https://spacy.io/universe/project/negspacy
            https://code.google.com/archive/p/negex/downloads?page=2
        """

        logging.info('\tInitialize negation (NegEx) features.')

        self.nlp = nlp

        if lang not in ['de', 'en']:
            raise ValueError('The language', lang, 'is not applicable for the negation module.')

        self.neg_keys = ['pseudo', 'preceding', 'following', 'termination']
        default_features = self.neg_keys + ['negated_entities']

        if features == 'default':
            self.features = default_features
        else:
            if set(features).intersection(default_features) == set():
                raise ValueError('Your negation features ' + ' '.join(self.features) + ' are not defined! Allowed definitions: ', default_features)
            else:
                logging.info()('\t\tDefined features:', features)
                self.features = features

        self.path_external_resources = os.path.join(os.sep.join(
            os.path.abspath(__file__).split(os.sep)[:-4]),
            'ext_res',
            'negex',
            lang
        )
        self.matcher = PhraseMatcher(self.nlp.vocab, attr='LOWER')
        self.d_paths = {}

        for file in glob.glob(self.path_external_resources + os.sep + '**/*.txt', recursive=True):
            file_name = file.replace(self.path_external_resources, '').replace('.txt', '').replace(os.sep, '')
            patterns = [self.nlp(entry) for entry in [item.rstrip() for item in open(file, 'r', encoding='utf8').readlines()]]
            self.matcher.add(file_name, None, *patterns)
            logging.info('\t\t' + str(len(patterns)) + ' [' + file_name + '] negation patterns added.')

        self.chunk_prefix = list(nlp.tokenizer.pipe(list()))

    def termination_boundaries(self, doc, terminating):
        """
        Create sub sentences based on terminations found in text.

        Parameters
        ----------
        doc: object
            spaCy Doc object
        terminating: list
            list of tuples with (match_id, start, end)

        returns
        -------
        boundaries: list
            list of tuples with (start, end) of spans

        """
        sent_starts = [sent.start for sent in doc.sents]
        terminating_starts = [t[1] for t in terminating]
        starts = sent_starts + terminating_starts + [len(doc)]
        starts.sort()
        boundaries = list()
        index = 0
        for i, start in enumerate(starts):
            if not i == 0:
                boundaries.append((index, start))
            index = start
        return boundaries

    def feat_doc(self, doc):

        preceding = list()
        following = list()
        terminating = list()

        matches = self.matcher(doc)

        negation_features = {key: len([doc[start:end] for match_id, start, end in matches if str(doc.vocab.strings[match_id]) == key]) for key in self.neg_keys}

        pseudo = [(match_id, start, end) for match_id, start, end in matches if self.nlp.vocab.strings[match_id] == "pseudo"]

        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "pseudo":
                continue
            pseudo_flag = False
            for p in pseudo:
                if p[1] <= start <= p[2]:
                    pseudo_flag = True
                    continue
            if not pseudo_flag:
                if self.nlp.vocab.strings[match_id] == "Preceding":
                    preceding.append((match_id, start, end))
                elif self.nlp.vocab.strings[match_id] == "Following":
                    following.append((match_id, start, end))
                elif self.nlp.vocab.strings[match_id] == "Termination":
                    terminating.append((match_id, start, end))
                else:
                    logging.warning(f"phrase {doc[start:end].text} not in one of the expected matcher types.")

        negated_entities = 0
        boundaries = self.termination_boundaries(doc, terminating)

        for b in boundaries:

            sub_preceding = [i for i in preceding if b[0] <= i[1] < b[1]]
            sub_following = [i for i in following if b[0] <= i[1] < b[1]]

            for e in doc[b[0] : b[1]].ents:
                if any(pre < e.start for pre in [i[1] for i in sub_preceding]):
                    negated_entities += 1
                    continue
                if any(fol > e.end for fol in [i[2] for i in sub_following]):
                    negated_entities += 1
                    continue
                if self.chunk_prefix:
                    if any(e.text.lower().startswith(c.text.lower()) for c in self.chunk_prefix):
                        negated_entities += 1

        negation_features['negated_entities'] = negated_entities

        data = {}

        if doc._.n_tokens != 0:
            data['features'] = {f: negation_features[f] / doc._.n_tokens for f in negation_features}
        else:
            data['features'] = {f: 0 for f in data['counts']}

        data['counts'] = negation_features

        return data
