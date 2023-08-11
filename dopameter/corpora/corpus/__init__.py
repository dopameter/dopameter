import collections
import sys
from collections import Counter
from dopameter import ConfLanguages


class Corpus:

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

    * characters : int
    * sentences_cnt : int
    * sentences : set
    * tokens_cnt : int

    * types : set
    * lemmata : set

    * documents : dict
    * document_cnt_characteristics : dict

    * features : {}
    * macro_features : {}
    * counts : {}
    * spans : {}

    * terminologies : {}
    * ngrams : {}
    * emotion : {}
    * surface : {}
    * lexical_richness : {}
    * syntax_dep : {}
    * syntax_const : {}

    * semantic_relations_wordnet : dict

    """

    def __init__(self, corpus_path, lang, name, files, encoding, features, external_resources):
        self.path = corpus_path
        self.lang = lang
        self.name = name
        self.files = files
        self.encoding = encoding

        self.characters = 0
        self.sentences_cnt = 0
        self.sentences = set()
        self.tokens_cnt = 0

        self.types = set()
        self.lemmata = set()

        self.documents = {} # for BLEU needed
        self.document_cnt_characteristics = {}

        self.features = {}
        self.macro_features = {}
        self.counts = {}
        self.spans = {}

        self.terminologies = {}
        self.ngrams = {}
        self.emotion = {}
        self.surface = {}
        self.lexical_richness = {}
        self.syntax_dep = {}
        self.syntax_const = {}

        self.semantic_relations_wordnet = {
            'doc': collections.Counter(),
            'sent': collections.Counter()
        }

        self.init_properties(features, external_resources)


    def update_count_characteristics(self, cnt, doc_name):
        """update counts during a count routine"""
        self.characters += cnt['characters_cnt']
        self.sentences_cnt += cnt['sentences_cnt']
        self.tokens_cnt += cnt['tokens_cnt']

        self.types.update(cnt['types'])
        self.lemmata.update(cnt['lemmata'])
        self.sentences.update(cnt['different_sentences'])

        self.document_cnt_characteristics[doc_name] = {}
        self.document_cnt_characteristics[doc_name]['characters']           = cnt['characters_cnt']
        self.document_cnt_characteristics[doc_name]['sentences']            = cnt['sentences_cnt']
        self.document_cnt_characteristics[doc_name]['different_sentences']  = len(cnt['different_sentences'])
        self.document_cnt_characteristics[doc_name]['tokens']               = cnt['tokens_cnt']
        self.document_cnt_characteristics[doc_name]['types']                = len(cnt['types'])
        self.document_cnt_characteristics[doc_name]['lemmata']              = len(cnt['lemmata'])


    def count_characteristics(self):
        """get count characteristics of a corpus"""
        return {
            'documents':            len(self.files),
            'sentences':            self.sentences_cnt,
            'different_sentences':  len(self.sentences),
            'tokens':               self.tokens_cnt,
            'types':                len(self.types),
            'characters':           self.characters,
            'lemmata':              len(self.lemmata)
        }

    def update_bleu_props(self, doc, doc_name):
        """update bleu documents during a corpus comparison"""
        self.documents[doc_name] = [tokens for tokens in doc._.tokens]

    def init_properties(self, features, external_resources):
        """initialize properties of a corpus - for corpus creation process"""
        if 'corpus_characteristics' in features.keys():
            self.features['corpus_characteristics'] = {}
            self.macro_features['corpus_characteristics'] = {}

        if 'pos' in features.keys():
            self.counts['pos'] = {}

        if 'ner' in features.keys():
            self.counts['ner'] = {}

        if 'ngrams' in features.keys():
            self.macro_features['n_grams'] = {}
            self.features['ngrams'] = {}
            self.counts['ngrams'] = {}

            for n in features['ngrams']:
                self.ngrams[n] = {}

        if 'token_characteristics' in features.keys():
            self.features['token_characteristics'] = {}
            self.counts['token_characteristics'] = {}
            self.spans['token_characteristics'] = {}

        if 'surface' in features.keys():
            self.features['surface'] = {}
            self.counts['surface'] = {}
            self.macro_features['surface'] = {}

            self.surface['token_len_chars'] = []
            self.surface['sent_len_tokens'] = []
            self.surface['sent_len_chars'] = []

            self.surface['toks_min_three_syllables'] = collections.Counter()
            self.surface['toks_larger_six_letters'] = collections.Counter()
            self.surface['toks_one_syllable'] = collections.Counter()

            self.surface['cnt_syllables'] = 0
            self.surface['cnt_words'] = 0
            self.surface['cnt_poly_syllables'] = 0
            self.surface['cnt_letter_tokens'] = 0
            self.surface['cnt_no_digit_tokens'] = 0

            self.surface['syllables_per_word'] = []
            self.surface['sent_lenghts'] = []
            self.surface['cnt_pos'] = Counter({})
            self.surface['segments'] = []
            self.surface['cnt_diff_words'] = 0

            self.surface['syllables'] = Counter({})
            self.surface['words_poly_syllables'] = Counter({})
            self.surface['letter_tokens'] = Counter({})
            self.surface['no_digit_tokens'] = Counter({})
            self.surface['sentences'] = Counter({})

        if 'lexical_richness' in features.keys():
            self.features['lexical_richness'] = {}
            self.counts['lexical_richness'] = {}
            self.macro_features['lexical_richness'] = {}

            self.lexical_richness['freq_list'] = collections.Counter({})
            self.lexical_richness['corpus_tokens'] = []
            self.lexical_richness['function_words'] = collections.Counter({})

        if 'syntax_dependency_metrics' in features.keys():
            self.features['syntax_dependency_metrics'] = {}
            self.syntax_dep['children_per_node'] = []
            self.syntax_dep['max_depth'] = []
            self.syntax_dep['dep_distance'] = []
            self.syntax_dep['out_degree_centralization'] = []
            self.syntax_dep['closeness_centrality'] = []

        if 'syntax_dependency_tree' in features.keys():
            self.features['syntax_dependency_tree'] = {}
            self.counts['syntax_dependency_tree'] = {}

        if 'syntax_constituency_metrics' in features.keys():
            self.features['syntax_constituency_metrics'] = {}
            self.syntax_const['tree_heights'] = []
            self.syntax_const['doc_out_degrees'] = []
            self.syntax_const['non_terminals_wo_leaves'] = []
            self.syntax_const['doc_const'] = []
            self.syntax_const['t_units'] = []
            self.syntax_const['out_degree_centralization'] = []
            self.syntax_const['closeness_centrality'] = []

        if 'syntax_constituency_tree' in features.keys():
            self.features['syntax_constituency_tree'] = {}
            self.counts['syntax_constituency_tree'] = {}


        if 'emotion' in features.keys():
            self.features['emotion'] = {}
            self.macro_features['emotion'] = {}
            self.emotion['features'] = {feat: 0 for feat in self.features['emotion']}
            self.emotion['sum_emotions_words'] = 0

        if ('wordnet_synsets' in features.keys() or
            'wordnet_senses' in features.keys() or
            'wordnet_semantic_relations' in features.keys()
        ):
            if self.lang not in ConfLanguages().wordnet_languages:
                sys.exit('WordNet is not configured or installed. Check your config file and your installation.')
            else:

                if 'wordnet_semantic_relations' in features.keys():
                    self.features['wordnet_semantic_relations'] = {}
                    self.macro_features['wordnet_semantic_relations'] = {}

                if 'wordnet_synsets' in features.keys() and self.lang in ConfLanguages().wordnet_languages:
                    self.features['wordnet_synsets'] = {}
                    self.counts['wordnet_synsets'] = {}
                    self.terminologies['wordnet_synsets'] = collections.Counter({})

                if 'wordnet_senses' in features.keys() and self.lang in ConfLanguages().wordnet_languages:
                    self.features['wordnet_senses'] = {}
                    self.counts['wordnet_senses'] = {}
                    self.terminologies['wordnet_senses'] = collections.Counter({})

        if 'dictionary_lookup' in features.keys():
            for feat in features['dictionary_lookup']:
                self.features['dictionary_lookup_' + feat] = {}
                self.counts['dictionary_lookup_' + feat] = {}

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

        if 'features' in data.keys():
            for key in data['features'].keys():
                if key not in self.features[feature].keys():
                    self.features[feature][key] = {}
                self.features[feature][key][doc_name] = data['features'][key]

            if feature == 'emotion':
                self.emotion['features'] = dict(Counter(self.emotion['features']) + Counter(data['emotion']['features']))
                if 'sum_emotions_words' in data['emotion'].keys():
                    self.emotion['sum_emotions_words'] += data['emotion']['sum_emotions_words']

            if feature == 'lexical_richness':
                if 'freq_list' in data['lexical_richness'].keys():
                    self.lexical_richness['freq_list'] += data['lexical_richness']['freq_list']
                if 'corpus_tokens' in data['lexical_richness'].keys():
                    self.lexical_richness['corpus_tokens'] += data['lexical_richness']['corpus_tokens']
                if 'function_words' in data['lexical_richness'].keys():
                    self.lexical_richness['function_words'].update(data['lexical_richness']['function_words'])

            if feature == 'syntax_dependency_metrics':
                self.syntax_dep['children_per_node'] += data['syntax_dep']['children_per_node']
                self.syntax_dep['max_depth'] += data['syntax_dep']['max_depth']
                self.syntax_dep['dep_distance'] += data['syntax_dep']['dep_distance']
                self.syntax_dep['out_degree_centralization'] += data['syntax_dep']['out_degree_centralization']
                self.syntax_dep['closeness_centrality'] += data['syntax_dep']['closeness_centrality']

            if feature == 'syntax_constituency_metrics':

                self.syntax_const['tree_heights'] += data['syntax_const']['tree_heights']
                self.syntax_const['doc_out_degrees'] += data['syntax_const']['doc_out_degrees']
                self.syntax_const['non_terminals_wo_leaves'] += data['syntax_const']['non_terminals_wo_leaves']
                self.syntax_const['doc_const'] += data['syntax_const']['doc_const']
                self.syntax_const['t_units'] += data['syntax_const']['t_units']
                self.syntax_const['out_degree_centralization'] += data['syntax_const']['out_degree_centralization']
                self.syntax_const['closeness_centrality'] += data['syntax_const']['closeness_centrality']

            if feature == 'surface':
                if 'token_len_chars' in data['surface'].keys():
                    self.surface['token_len_chars'] += data['surface']['token_len_chars']
                if 'sent_len_tokens' in data['surface'].keys():
                    self.surface['sent_len_tokens'] += data['surface']['sent_len_tokens']
                if 'sent_len_chars' in data['surface'].keys():
                    self.surface['sent_len_chars'] += data['surface']['sent_len_chars']
                if 'toks_min_three_syllables' in data['surface'].keys():
                    self.surface['toks_min_three_syllables'].update(data['surface']['toks_min_three_syllables'])
                if 'toks_larger_six_letters' in data['surface'].keys():
                    self.surface['toks_larger_six_letters'].update(data['surface']['toks_larger_six_letters'])
                if 'toks_one_syllable' in data['surface'].keys():
                    self.surface['toks_one_syllable'].update(data['surface']['toks_one_syllable'])

                if 'cnt_syllables' in data['surface'].keys():
                    self.surface['cnt_syllables'] += data['surface']['cnt_syllables']
                if 'cnt_words' in data['surface'].keys():
                    self.surface['cnt_words'] += data['surface']['cnt_words']
                if 'cnt_poly_syllables' in data['surface'].keys():
                    self.surface['cnt_poly_syllables'] += data['surface']['cnt_poly_syllables']
                if 'cnt_letter_tokens' in data['surface'].keys():
                    self.surface['cnt_letter_tokens'] += data['surface']['cnt_letter_tokens']
                if 'cnt_no_digit_tokens' in data['surface'].keys():
                    self.surface['cnt_no_digit_tokens'] += data['surface']['cnt_no_digit_tokens']

                if 'cnt_pos' in data['surface'].keys():
                    self.surface['cnt_pos'] += data['surface']['cnt_pos']
                if 'segments' in data['surface'].keys():
                    self.surface['segments'].extend(data['surface']['segments'])

                if 'syllables' in data['surface'].keys():
                    self.surface['syllables'].update(data['surface']['syllables'])
                if 'letter_tokens' in data['surface'].keys():
                    self.surface['letter_tokens'].update(data['surface']['letter_tokens'])
                if 'no_digit_tokens' in data['surface'].keys():
                    self.surface['no_digit_tokens'].update(data['surface']['no_digit_tokens'])
                if 'sentences' in data['surface'].keys():
                    self.surface['sentences'].update(data['surface']['sentences'])

                if self.lang == 'de':
                    if 'syllables_per_word' in data['surface'].keys():
                        self.surface['syllables_per_word'] += data['surface']['syllables_per_word']
                    if 'sent_lenghts' in data['surface'].keys():
                        self.surface['sent_lenghts'] += data['surface']['sent_lenghts']
                    if 'toks_larger_six_letters' in data['surface'].keys():
                        self.surface['toks_larger_six_letters'] += data['surface']['toks_larger_six_letters']

                if self.lang == 'en':
                    if 'cnt_diff_words' in self.surface.keys():
                        self.surface['cnt_diff_words'] += data['surface']['cnt_diff_words']

            if feature == 'wordnet_semantic_relations':
                self.semantic_relations_wordnet['doc'] += data['doc']

        if feature == 'wordnet_synsets':
            self.terminologies['wordnet_synsets'].update(data['counts'])

        if feature == 'wordnet_senses':
            self.terminologies['wordnet_senses'].update(data['counts'])

        if feature == 'ngrams':
            for n in data['ngrams'].keys():
                if n != 'counts':
                    self.ngrams[n].update(data['ngrams'][n])
