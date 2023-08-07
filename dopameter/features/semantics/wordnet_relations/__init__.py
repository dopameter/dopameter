import collections
import json
import logging
import os

import numpy as np
import wn

from dopameter.configuration.installation import ConfLanguages


def handle_wordnet(lang):

    """

    Parameters
    ----------
    lang : basestring

    Returns
    -------
    dict
        dictionary with transformed structure of semantic networks of the wn package resources
    """

    wordnet_object = wn.Wordnet(ConfLanguages().wordnet_languages[lang])

    _wordnet_properties = {
        'synsets': [syn.id for syn in wordnet_object.synsets()],
        'senses': [sen.id for sen in wordnet_object.senses()]
    }

    for token_type in [word.lemma() for word in wordnet_object.words()]:

        synset_list = wordnet_object.synsets(token_type)

        synonyms = set()
        hypernyms = set()
        antonyms = set()
        max_depths = []
        min_depths = []

        _wordnet_properties[token_type] = {}
        _wordnet_properties[token_type]['synsets'] = [syn.id for syn in synset_list]
        _wordnet_properties[token_type]['senses'] = [sen.id for sen in wordnet_object.senses(token_type)]

        for synset in synset_list:

            synoynms = set(synset.lemmas())
            if token_type in synoynms:
                synoynms.remove(token_type)
            if synoynms:
                synonyms.update(synoynms)

            if synset.get_related('antonym'):
                antonyms.update([lemma for ant in synset.get_related('antonym') for lemma in ant.lemmas()])

            for sense in synset.senses():
                if sense.get_related('antonym'):
                    ant = sense.get_related('antonym')
                    for a in ant:
                        antonyms.update(a.word().lemma())

            max_depths.append(synset.max_depth())
            min_depths.append(synset.min_depth())

        if synonyms:
            _wordnet_properties[token_type]['synonyms'] = list(synonyms)

        hyponyms = set([w for synset in synset_list for syn in synset.hyponyms() for word in syn.words() for w in word.forms()])
        if hyponyms:
            _wordnet_properties[token_type]['hyponyms'] = list(hyponyms)

        hyponyms = set([w for synset in synset_list for syn in synset.hypernyms() for word in syn.words() for w in word.forms()])
        if hypernyms:
            _wordnet_properties[token_type]['hypernyms'] = list(hypernyms)

        taxonyms = set(hyponyms)
        taxonyms.update(set(hypernyms))
        if taxonyms:
            _wordnet_properties[token_type]['taxonyms'] = list(taxonyms)

        holonyms = set([w for synset in synset_list for syn in synset.holonyms() for word in syn.words() for w in word.forms()])
        if holonyms:
            _wordnet_properties[token_type]['holonyms'] = list(holonyms)

        meronyms = set([w for synset in synset_list for syn in synset.meronyms() for word in syn.words() for w in word.forms()])
        if meronyms:
            _wordnet_properties[token_type]['meronyms'] = list(meronyms)

        if antonyms:
            _wordnet_properties[token_type]['antonyms'] = list(antonyms)

        if max_depths:
            _wordnet_properties[token_type]['max_depths'] = max_depths
        if min_depths:
            _wordnet_properties[token_type]['min_depths'] = min_depths

    return _wordnet_properties

def filter_span(span):
    """filter tokens to spare resource requests

    Parameters
    ----------
    span : spaCy span (or Doc)

    Returns
    -------
        collection with counted filtered tokens
    """
    return collections.Counter(
        [token.lemma_ for token in span if
                (not token.is_punct) and (not token.like_url) and (not token.like_email) and
                (not token.is_bracket) and (not token.is_quote) and (not token.is_stop) and (not token.is_space)]
        )

class WordNetFeatures:
    """Get Semantic Relations of WordNet Resources

    Parameters
    ----------
    lang : basestring
    features : list, [
                'wordnet_synsets': 'default',
                'wordnet_senses': 'default',
                'wordnet_semantic_relations': 'default'
        ]

    Attributes
    ----------
    * This package is derived into 2 parts:
        * 'wordnet_synsets': occurrences of synsets / nodes from wordnet
        * 'wordnet_senses': occurrences of senses / nodes from wordnet
        * 'wordnet_semantic_relations': Metrics / Scores
            * `sem_depth_min`: the minimal path length of each reading of each lemma within in a document (distance from the top node of the semantic network to the lemma) following taxonomic links (hypernym or hyponymy links, only), sum up these individual length scores and average over the number of all the lemmas' readings.
            * `sem_depth_max`: the maximal path length of each reading of each lemma within in a document (distance from the top node of the semantic network to the lemma) following taxonomic links (hypernym or hyponymy links, only), sum up these individual length scores and average over the number of all the lemmas' readings.
            * `sem_rich_*`: For each (reading of the) lemma in a sentence, we determine all of its semantic relation instances (i.e., hypernyms, hyponyms, parts (is-part) and wholes (has-part), antonyms) it shares with other lemmas in the lexicon and average this number over all readings per lemma and all sentences in the document. We also supply semantic richness scores for each specific semantic relation (i.e., sem_rich_hypernyms, ..., sem_rich_antonyms).
            * `synsets_avg` : average amount of synsets in a document / corpus
            * `senses_avg` : average amount of senses in a document / corpus

    Notes
    -----

    define in configuration .json file under features:

    * default:
       "tasks": ["features", "counts"],
       "features": {
         "wordnet_synsets": "default",
         "wordnet_senses": "default",
         "wordnet_semantic_relations": "default"
       }
    * or in detail:
       "tasks": ["features", "counts"],
       "features":
       {
         "wordnet_semantic_relations":
            [
              'sem_rich_hypernyms',
              'sem_rich_hyponyms',
              'sem_rich_taxonyms',
              'sem_rich_antonyms',
              'sem_rich_synonyms',
              'sem_rich_meronyms',
              'sem_rich_holonyms',
              'sem_rich',
              'sem_depth_min',
              'sem_depth_max',
              'synsets_avg',
              'senses_avg'
            ],
         "wordnet_synsets": ["odenet-11460-n", "odenet-3279-a, "odenet-15368-a"], # only examples, for the full list, check the docuemntation of GermaNet
         "wordnet_senses": ["w45444_11460-n", "w15016_3279-a", "w15016_15368-a"], # only examples, for the full list, check the docuemntation of GermaNet
       }
    """

    def __init__(
            self,
            lang,
            features={
                'wordnet_synsets': 'default',
                'wordnet_senses': 'default',
                'wordnet_semantic_relations': 'default'
            }
    ):
        logging.info('\tInitialize WordNet features.')

        wordnet_langs = ConfLanguages().wordnet_languages
        inst_langs = [lex.language for lex in wn.lexicons()]
        if lang not in inst_langs:
            if lang in wordnet_langs:
                    raise ValueError('The language', lang, 'is not installed. Install the WordNet (wn) language modul via language installation instructions.')
            else:
                raise ValueError('The language', lang, 'is not available for semantic features by WordNet.')

        self.path_extract_wordnet = os.path.join(
            os.sep.join(os.path.abspath(__file__).split(os.sep)[:-5]),
            'ext_res',
            'semantics',
            'wordnet_data_' + lang + '.json'
        )

        if not os.path.exists(self.path_extract_wordnet):
            self.wn_data = handle_wordnet(lang=lang)

            if not os.path.isdir(os.path.basename(self.path_extract_wordnet)):
                os.mkdir(os.path.basename(self.path_extract_wordnet))

            with open(
                    file=self.path_extract_wordnet,
                    mode='w',
                    encoding='utf-8'
            ) as f:
                json.dump(self.wn_data, f, ensure_ascii=False)
        else:
            with open(self.path_extract_wordnet) as f:
                self.wn_data = json.load(f)

        default_features = {
            'wordnet_synsets':  self.wn_data['synsets'],
            'wordnet_senses':   self.wn_data['senses'],
            'wordnet_semantic_relations': [
                'sem_rich_hypernyms',
                'sem_rich_hyponyms',
                'sem_rich_taxonyms',
                'sem_rich_antonyms',
                'sem_rich_synonyms',
                'sem_rich_meronyms',
                'sem_rich_holonyms',
                'sem_rich',
                'sem_depth_min',
                'sem_depth_max',
                'synsets_avg',
                'senses_avg'
            ]
        }

        self.features = {}

        for feat in default_features:
            if feat in features.keys():
                if features[feat] == 'default':
                    self.features[feat] = default_features[feat]
                else:
                    if set(features[feat]).intersection(default_features[feat]) == set():
                        raise ValueError('Your syntax WordNet feautres ' + ' '.join(features[feat]) + ' are not defined! Allowed definitions: ', default_features[feat])
                    else:
                        logging.info('\t\tDefined features: ' + features)
                        self.features[feat] = features[feat]


    def compute_semantic_relations_span(self, filtered_span, doc_len):
        """
        Computes semantic relations of a given filtered span

        Parameters
        ----------
        filtered_span : dictionary (created via collections.Count)
        doc_len : integer

        Returns
        -------
        dict
            dictionary with properties of semantic relation scores.
        """

        min_depths = []
        max_depths = []
        synsets = []
        senses = []

        sem_rel = collections.defaultdict(int)

        for token in filtered_span:
            if token in self.wn_data.keys():
                if 'synonyms' in self.wn_data[token].keys():
                    sem_rel['sem_rich_synonyms'] += filtered_span[token] * len(self.wn_data[token]['synonyms'])
                if 'hypernyms' in self.wn_data[token].keys():
                    sem_rel['sem_rich_hypernyms'] += filtered_span[token] * len(self.wn_data[token]['hypernyms'])
                if 'hyponyms' in self.wn_data[token].keys():
                    sem_rel['sem_rich_hyponyms'] += filtered_span[token] * len(self.wn_data[token]['hyponyms'])
                if 'taxonyms' in self.wn_data[token].keys():
                    sem_rel['sem_rich_taxonyms'] += filtered_span[token] * len(self.wn_data[token]['taxonyms'])
                if 'meronyms' in self.wn_data[token].keys():
                    sem_rel['sem_rich_meronyms'] += filtered_span[token] * len(self.wn_data[token]['meronyms'])
                if 'holonyms' in self.wn_data[token].keys():
                    sem_rel['sem_rich_holonyms'] += filtered_span[token] * len(self.wn_data[token]['holonyms'])
                if 'antonyms' in self.wn_data[token].keys():
                    sem_rel['sem_rich_antonyms'] += filtered_span[token] * len(self.wn_data[token]['antonyms'])
                if 'semantic_relations' in self.wn_data[token].keys():
                    sem_rel['sem_rich'] += filtered_span[token] * len(self.wn_data[token]['semantic_relations'])

                if 'min_depths' in self.wn_data[token].keys():
                    min_depths += self.wn_data[token]['min_depths']

                if 'max_depths' in self.wn_data[token].keys():
                    max_depths += self.wn_data[token]['max_depths']

                synsets += self.wn_data[token]['synsets']
                senses += self.wn_data[token]['senses']

        if doc_len != 0:
            sem_rel = {k: sem_rel[k] / doc_len for k in sem_rel.keys()}
        else:
            sem_rel = {k: 0 for k in sem_rel.keys()}

        if min_depths:
            sem_rel['sem_depth_min'] = np.mean(min_depths)
        else:
            sem_rel['sem_depth_min'] = 0

        if min_depths:
            sem_rel['sem_depth_max'] = np.mean(max_depths)
        else:
            sem_rel['sem_depth_max'] = 0

        if doc_len != 0:
            sem_rel['synsets_avg'] = len(synsets) / doc_len
            sem_rel['senses_avg'] = len(senses) / doc_len
        else:
            sem_rel['synsets_avg'] = 0
            sem_rel['senses_avg'] = 0
        return sem_rel


    def feat_doc(self, doc):
        """
        Get metrics and counts for a document of semantic relations

        Parameters
        ----------
        doc : spaCy Doc

        Returns
        -------
        dict
            dictionary with three parts of (1) wordnet_semantic_relations 'features' as key including the metrics and 'wordnet_semantic_relations' with interim results and 'counts' including amount of countable metrics such as (2) 'wordnet_synsets' and (3) 'wordnet_senses' with count occurrences and (4) 'doc' with interim resluts of metrics
        """

        filtered_doc = filter_span(doc)
        data = {}

        if 'wordnet_semantic_relations' in self.features.keys():
            sem_rel_doc = self.compute_semantic_relations_span(
                filtered_span=filtered_doc,
                doc_len=doc._.n_tokens
            )

            data['wordnet_semantic_relations'] = {}
            data['wordnet_semantic_relations']['features'] = sem_rel_doc
            data['wordnet_semantic_relations']['doc'] = collections.Counter(sem_rel_doc)

        if 'wordnet_synsets' in self.features.keys():
            synsets = collections.Counter([i for syn in [self.wn_data[k]['synsets'] * filtered_doc[k] for k in filtered_doc if k in self.wn_data.keys()] for i in syn])
            data['wordnet_synsets'] = {}
            data['wordnet_synsets']['counts'] = synsets

        if 'wordnet_senses' in self.features.keys():
            synsets = collections.Counter([i for syn in [self.wn_data[k]['senses'] * filtered_doc[k] for k in filtered_doc if k in self.wn_data.keys()] for i in syn])
            data['wordnet_senses'] = {}
            data['wordnet_senses']['counts'] = synsets

        return data

    def feat_corpus(self, corpus):
        """Get metrics of corpus wise semantic relations (by Wordnets

        Parameters
        ----------
        corpus

        Returns
        -------
            dictionary with features of semantic relations
        """

        return {'features': dict({k: corpus.semantic_relations_wordnet['doc'][k] / corpus.tokens_cnt for k in corpus.semantic_relations_wordnet['doc']})}