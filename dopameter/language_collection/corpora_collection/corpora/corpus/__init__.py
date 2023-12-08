import sys
from dopameter.configuration.installation import ConfLanguages
from dopameter.language_collection.corpora_collection import CorporaCollection
from dopameter.language_collection.corpora_collection.corpora.corpus.resources import Resources
from dopameter.language_collection.corpora_collection.corpora.corpus.sizes import Sizes


class Corpus(CorporaCollection):

    """
    Corpus: creates and update corpora

    Parameters
    ----------

    corpus_path : str
    lang : str
    name : str
    files : list of str
    encoding : str
    features : dict
    external_resources : dict

    Attributes
    ----------
    * path : str
    * lang : str
    * name : str
    * files : list of str
    * encoding : str

    * documents : dict
    * document_cnt_characteristics : dict

    * features : {}
    * macro_features : {}
    * counts : {}

    """

    def __init__(self, corpus_path, lang, name, files, encoding, features, external_resources, collection_name):
        super().__init__(lang, collection_name, features)

        self.path = corpus_path
        self.lang = lang
        self.name = name
        self.files = files
        self.encoding = encoding
        self.collection_name = collection_name

        self.documents = {}
        self.documents_pos = {}
        self.document_cnt_characteristics = {}

        self.init_corpus_properties(features, external_resources)


    def init_corpus_properties(self, features, external_resources):
        """initialize properties of a corpus - for corpus creation process"""
        self.features, self.macro_features, self.counts, self.resources, self.sizes = init_scores(conf_features=features, lang=self.lang)

    def count_characteristics(self):
        """get count characteristics of a corpus"""
        return {
            'documents':            len(self.files),

            'sentences':            self.sizes.sentences_cnt,
            'different_sentences':  len(self.sizes.sentences),

            'tokens':               self.sizes.tokens_cnt,
            'types':                len(self.sizes.types),
            'lemmata':              len(self.sizes.lemmata),

            'characters':           self.sizes.characters
        }

    def update_documents(self, doc, doc_name):
        """update a corpus with documents"""
        self.documents[doc_name] = [tokens for tokens in doc._.tokens]

        if 'pos_tags' in doc._.__dict__['_extensions'].keys():
            self.documents_pos[doc_name] = [tokens for tokens in doc._.pos_tags]

    def update_properties(self, feature, data, doc_name):
        """update properties of a corpus by single documents - for corpus creation process

        Parameters
        ----------
        feature : str
        data : dict
        doc_name : str
        """

        if 'counts' in data.keys():
            for key in data['counts'].keys():
                if key not in self.counts[feature].keys():
                    self.counts[feature][key] = {}
                self.counts[feature][key][doc_name] = data['counts'][key]

        if 'features' in data.keys() and feature != 'basic_counts':
            for key in data['features'].keys():
                if key not in self.features[feature].keys():
                    self.features[feature][key] = {}
                self.features[feature][key][doc_name] = data['features'][key]

        self.resources.update_resource_by_data(feature=feature, data=data)

        if 'basic_counts' == feature:
            self.sizes.update_sizes_by_data(data)

            self.document_cnt_characteristics[doc_name] = {}
            self.document_cnt_characteristics[doc_name]['characters'] = data['characters_cnt']
            self.document_cnt_characteristics[doc_name]['sentences'] = data['sentences_cnt']
            self.document_cnt_characteristics[doc_name]['different_sentences'] = len(data['different_sentences'])
            self.document_cnt_characteristics[doc_name]['tokens'] = data['tokens_cnt']
            self.document_cnt_characteristics[doc_name]['types'] = len(data['types'])
            self.document_cnt_characteristics[doc_name]['lemmata'] = len(data['lemmata'])

    def clear(self):
        self.resources.clear()
        self.counts.clear()
        self.features.clear()

def init_scores(conf_features, lang):
    """initialize properties of a corpus - for corpus creation process"""

    features = {}
    macro_features = {}
    counts = {}

    if 'corpus_characteristics' in conf_features.keys():
        features['corpus_characteristics'] = {}
        macro_features['corpus_characteristics'] = {}

    if 'token_characteristics' in conf_features.keys():
        features['token_characteristics'] = {}
        counts['token_characteristics'] = {}

    if 'pos' in conf_features.keys():
        counts['pos'] = {}

    if 'ner' in conf_features.keys():
        counts['ner'] = {}

    if 'ngrams' in conf_features.keys():
        features['ngrams'] = {}
        counts['ngrams'] = {}
        macro_features['ngrams'] = {}

    if 'lexical_diversity' in conf_features.keys():
        features['lexical_diversity'] = {}
        counts['lexical_diversity'] = {}
        macro_features['lexical_diversity'] = {}

    if 'surface' in conf_features.keys():
        features['surface'] = {}
        counts['surface'] = {}
        macro_features['surface'] = {}

    if 'syntax_dependency_metrics' in conf_features.keys():
        features['syntax_dependency_metrics'] = {}
        macro_features['syntax_dependency_metrics'] = {}

    if 'syntax_dependency_tree' in conf_features.keys():
        features['syntax_dependency_tree'] = {}
        counts['syntax_dependency_tree'] = {}

    if 'syntax_constituency_metrics' in conf_features.keys():
        features['syntax_constituency_metrics'] = {}
        macro_features['syntax_constituency_metrics'] = {}

    if 'syntax_constituency_tree' in conf_features.keys():
        features['syntax_constituency_tree'] = {}
        counts['syntax_constituency_tree'] = {}

    if set(conf_features.keys()).intersection({'wordnet_synsets', 'wordnet_senses', 'wordnet_semantic_relations'}):
        if lang not in ConfLanguages().wordnet_languages:
            sys.exit('WordNet is not configured or installed. Check your config file and your installation.')
        else:
            if 'wordnet_semantic_relations' in conf_features.keys():
                features['wordnet_semantic_relations'] = {}
                macro_features['wordnet_semantic_relations'] = {}

            for feat in (set(conf_features.keys())).intersection({'wordnet_synsets', 'wordnet_senses'}):
                features[feat] = {}
                counts[feat] = {}

    if 'dictionary_lookup' in conf_features.keys():
        for feat in conf_features['dictionary_lookup']:
            features['dictionary_lookup_' + feat] = {}
            counts['dictionary_lookup_' + feat] = {}

    if 'emotion' in conf_features.keys():
        features['emotion'] = {}
        macro_features['emotion'] = {}

    return features, macro_features, counts, Resources(conf_features=conf_features), Sizes()