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

    wordnet_languages = {
        "en": "omw-en:1.4",
        "de": "odenet:1.4"
    }

    wordnet_object = wn.Wordnet(wordnet_languages[lang])

    _wordnet_properties = {
        'synsets': [syn.id for syn in wordnet_object.synsets()],
        'senses': [sen.id for sen in wordnet_object.senses()],
        'entries': {}
    }

    for token_type in [word.lemma() for word in wordnet_object.words()]:

        synset_list = wordnet_object.synsets(token_type)

        antonyms = set()
        max_depths = []
        min_depths = []

        _wordnet_properties['entries'][token_type] = {}

        _wordnet_properties['entries'][token_type]['synsets'] = [syn.id for syn in synset_list]
        _wordnet_properties['entries'][token_type]['synsets_len'] = len([syn.id for syn in synset_list])

        _wordnet_properties['entries'][token_type]['senses'] = [sen.id for sen in wordnet_object.senses(token_type)]
        _wordnet_properties['entries'][token_type]['senses_len'] = len([sen.id for sen in wordnet_object.senses(token_type)])

        _wordnet_properties['entries'][token_type]['tok_length'] = len(token_type.split(' '))
        _wordnet_properties['entries'][token_type]['sem_rich'] = {'sem_rich':0}

        for synset in synset_list:

            synonyms = set(synset.lemmas())
            if token_type in synonyms:
                synonyms.remove(token_type)
            if synonyms:
                synonyms.update(synonyms)

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
            _wordnet_properties['entries'][token_type]['synonyms'] = list(synonyms)
            _wordnet_properties['entries'][token_type]['sem_rich']['synonyms'] = len(list(synonyms))
            _wordnet_properties['entries'][token_type]['sem_rich']['sem_rich'] += len(list(synonyms))

        hyponyms = set([w for synset in synset_list for syn in synset.hyponyms() for word in syn.words() for w in word.forms()])
        if hyponyms:
            _wordnet_properties['entries'][token_type]['hyponyms'] = list(hyponyms)
            _wordnet_properties['entries'][token_type]['sem_rich']['hyponyms'] = len(list(hyponyms))
            _wordnet_properties['entries'][token_type]['sem_rich']['sem_rich'] += len(list(hyponyms))

        hypernyms = set([w for synset in synset_list for syn in synset.hypernyms() for word in syn.words() for w in word.forms()])
        if hypernyms:
            _wordnet_properties['entries'][token_type]['hypernyms'] = list(hypernyms)
            _wordnet_properties['entries'][token_type]['sem_rich']['hypernyms'] = len(list(hypernyms))
            _wordnet_properties['entries'][token_type]['sem_rich']['sem_rich'] += len(list(hypernyms))

        taxonyms = set(hyponyms)
        taxonyms.update(set(hypernyms))
        if taxonyms:
            _wordnet_properties['entries'][token_type]['taxonyms'] = list(taxonyms)
            _wordnet_properties['entries'][token_type]['sem_rich']['taxonyms'] = len(list(taxonyms))

        holonyms = set([w for synset in synset_list for syn in synset.holonyms() for word in syn.words() for w in word.forms()])
        if holonyms:
            _wordnet_properties['entries'][token_type]['holonyms'] = list(holonyms)
            _wordnet_properties['entries'][token_type]['sem_rich']['holonyms'] = len(list(holonyms))
            _wordnet_properties['entries'][token_type]['sem_rich']['sem_rich'] += len(list(holonyms))

        meronyms = set([w for synset in synset_list for syn in synset.meronyms() for word in syn.words() for w in word.forms()])
        if meronyms:
            _wordnet_properties['entries'][token_type]['meronyms'] = list(meronyms)
            _wordnet_properties['entries'][token_type]['sem_rich']['meronyms'] = len(list(meronyms))
            _wordnet_properties['entries'][token_type]['sem_rich']['sem_rich'] += len(list(meronyms))

        if antonyms:
            _wordnet_properties['entries'][token_type]['antonyms'] = list(antonyms)
            _wordnet_properties['entries'][token_type]['sem_rich']['antonyms'] = len(list(antonyms))
            _wordnet_properties['entries'][token_type]['sem_rich']['sem_rich'] += len(list(antonyms))

        if max_depths:
            _wordnet_properties['entries'][token_type]['max_depths'] = max_depths
            _wordnet_properties['entries'][token_type]['sem_rich']['max_depths'] = len(max_depths)
        if min_depths:
            _wordnet_properties['entries'][token_type]['min_depths'] = min_depths
            _wordnet_properties['entries'][token_type]['sem_rich']['min_depths'] = len(min_depths)

    return _wordnet_properties


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

        self.ext_dict = collections.defaultdict(list)
        for elem in self.wn_data['entries']:
            dict_words = elem.split(' ')
            self.ext_dict[dict_words[0]] += [dict_words]


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


        semantic_relations = collections.defaultdict(int)

        synsets = []
        senses = []

        for i, token in enumerate(doc):
            if token.lemma_ in self.ext_dict.keys():
                for ent in self.ext_dict[token.lemma_]:

                    #if not token.is_stop:

                    if ent == [t.lemma_ for t in doc[i:(i + len(ent))]]:

                        hint = doc[i:(i + len(ent))].lemma_


                        if hint in self.wn_data['entries'].keys():
                            for item in self.wn_data['entries'][hint]['sem_rich']:
                                semantic_relations[item] += self.wn_data['entries'][hint]['sem_rich'][item]

                            synsets += self.wn_data['entries'][hint]['synsets']
                            semantic_relations['synsets_len'] += int(self.wn_data['entries'][hint]['synsets_len'])

                            senses += self.wn_data['entries'][hint]['senses']
                            semantic_relations['senses_len'] += int(self.wn_data['entries'][hint]['senses_len'])

        metrics = {}
        for item in semantic_relations:
            if item in ['hypernyms', 'hyponyms', 'taxonyms', 'antonyms', 'synonyms', 'meronyms', 'holonyms']:
                metrics['sem_rich_' + item] = semantic_relations[item] / doc._.n_tokens

            elif item == 'synsets_len':
                metrics['synsets_avg'] = semantic_relations[item] / doc._.n_tokens

            elif item == 'senses_len':
                metrics['senses_avg'] = semantic_relations[item] / doc._.n_tokens
            else:
                metrics[item] = semantic_relations[item] / doc._.n_tokens

        data = {}

        if 'wordnet_semantic_relations' in self.features.keys():
            data['wordnet_semantic_relations'] = {}
            data['wordnet_semantic_relations']['features'] = metrics
            data['wordnet_semantic_relations']['doc'] = semantic_relations

        if 'wordnet_synsets' in self.features.keys():
            data['wordnet_synsets'] = {'counts': collections.Counter(synsets)}

        if 'wordnet_senses' in self.features.keys():
            data['wordnet_senses'] = {'counts': collections.Counter(senses)}

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

        metrics = {}

        for item in corpus.resources.semantic_relations_wordnet:
            if item in ['hypernyms', 'hyponyms', 'taxonyms', 'antonyms', 'synonyms', 'meronyms', 'holonyms']:
                metrics['sem_rich_' + item] = corpus.resources.semantic_relations_wordnet[item] / corpus.sizes.tokens_cnt

            elif item == 'synsets_len':
                metrics['synsets_avg'] = corpus.resources.semantic_relations_wordnet[item] / corpus.sizes.tokens_cnt

            elif item == 'senses_len':
                metrics['senses_avg'] = corpus.resources.semantic_relations_wordnet[item] / corpus.sizes.tokens_cnt
            else:
                metrics[item] = corpus.resources.semantic_relations_wordnet[item] / corpus.sizes.tokens_cnt

        return {'features': metrics}
