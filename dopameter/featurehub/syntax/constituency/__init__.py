import collections
import sys
import logging
from nltk import ParentedTree
import numpy as np
import networkx as nx
import spacy  # do not delete this line!
import benepar # do not delete this line!
#os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from dopameter.configuration.installation import ConfLanguages


class ConstituencyFeatures:
    """Get Syntax Dependency Metrics

    uses Berkeley Neural Parser / python package 'benepar':
    * https://github.com/nikitakit/self-attentive-parser
    * https://pypi.org/project/benepar/

    Parameters
    ----------
    lang : string,
    features : list, default={
                        'syntax_constituency_metrics' : 'default',
                        'syntax_constituency_tree': 'default'
                        }

    Attributes
    ----------
    * This package is derived into 2 parts:
        * 'syntax_constituency_metrics` : Metrics / Scores
        * 'syntax_constituency_tree` : occurrences of parse node trees
    * `syntax_constituency_metrics` :
        * `AvgMaxDepth`: the maximum depth for each constituency parse tree, i.e., the longest path from the root node to a leaf node,
            is determined for each sentence, summed up for all sentences of a document and averaged over all sentences,
            to find the maximum depth (height) of the dependency tree by its root
        * `MaxFan`: the maximum fan-out of the constituency parse tree, i.e., the highest number of directly dominated nodes
            from an ancestor node in the entire parse tree, is determined over all sentences, and the maximum of all
            sentences of the document is selected
        * `AvgFan`: the average fan-out of the constituency parse tree, i.e., the highest number of directly dominated nodes
            from an ancestor node in the entire parse tree, is determined over all sentences, and the maximum of all
            sentences of the document is selected
        * `AvgNonTerminales_sent` : average amount of non terminal symbols without leaves of constituency trees
        * `AvgConstituents_sent` : average amount of constituents per sentence of all sentences divided by the amount of all sentences of a document
        * `AvgLenConstituents` : amount of the length of all constituents of a document divided by the amount of all constituents of a document
        * `AvgTunits_sent` : average amount of t-units divided by the amount of sentences per documents
        * `AvgLenTunits` : amount of all t-units divided by the amount of all t-units
        * `AvgOutdegreeCentralization` : avergage of outdegree centralization value of all sentences of document
            * computed over all dependency graphs over single sentences of the document)
            * value in {0,1};
            * if value == 1: all other nodes are dependent on root node and height is 1
            * computed by out-degrees values of all n nodes (count of child nodes per single node),
                maximal out-degree
        * `AvgClosenessCentralization`: avergage closeness centralization of all all dependency graphs of sentences of
        document;
            * closeness centralization is calculated as the reciprocal of the sum of the length of
            the distance (shortest paths) between two nodes x and y and x (d(x,y)), y in 1..n of a graph with n nodes
            (Bavelas 1950): (computed over all dependency graphs over single sentences of the document);
            * per default 0 if count of nodes is 1
    * `syntax_constituency_tree` :
        * occurrences of parse node trees
        * depends on language model, examples for German Language: 'nk', 'sb', 'ag', 'ROOT'; examples for English Language: 'acl', 'conj', 'xcomp'

    Notes
    -----

    define in configuration .json file under features:

    * default:
      "tasks": ["features", "counts"],
      "features": {
        "syntax_constituency_metrics" : "default",
        "syntax_constituency_metrics": "default"
      }
    * or in detail:
        "tasks": ["features", "counts"],
        "features": {
        "syntax_constituency_metrics" : [
            'AvgMaxDepth',
            'AvgFan',
            'MaxFan',
            'AvgNonTerminales_sent',
            'AvgConstituents_sent',
            'AvgTunits_sent',
            'AvgLenConstituents',
            'AvgLenTunits',
            'AvgOutdegreeCentralization',
            'AvgClosenessCentralization'
            ],
    "syntax_constituency_tree": ["ROOT", "CAP", "CS", "VP"], # only three examples
    }
    """

    def __init__(
            self,
            lang,
            features={
            'syntax_constituency_metrics' : 'default',
            'syntax_constituency_tree': 'default'
        }
    ):
        logging.info('\tInitialize syntax constituency features.')

        self.lang = lang
        self.benepar_langs = ConfLanguages().benepar_languages
        if self.lang in self.benepar_langs.keys():

            logging.info('\t\tload model ' + self.benepar_langs[self.lang])
            self.const_nlp = spacy.load(ConfLanguages().spacy_languages[lang])
            self.const_nlp.add_pipe("benepar", config={"model": self.benepar_langs[self.lang]})

        else:
            raise ValueError('The language', self.lang, ' is not available.',
                             'Check, if ', self.lang, ' is available and install it before running.',
                             'The constituency_count is not computed.')

        default_features = {
            'syntax_constituency_metrics': [
                'AvgMaxDepth',
                'AvgFan',
                'MaxFan',
                'AvgNonTerminales_sent',
                'AvgConstituents_sent',
                'AvgTunits_sent',
                'AvgLenConstituents',
                'AvgLenTunits',
                'AvgOutdegreeCentralization',
                'AvgClosenessCentralization'
            ],
            'syntax_constituency_tree': 'default' # Defaults ziehen!
        }

        self.features = {}

        if 'syntax_constituency_metrics' in features.keys():
            if features['syntax_constituency_metrics'] == 'default':
                self.features['syntax_constituency_metrics'] = default_features['syntax_constituency_metrics']
            else:
                if set(features).intersection(default_features) == set():
                    raise ValueError('Your syntax constituency ' + ' '.join(self.features) + ' are not defined! Allowed definitions: ', default_features)
                else:
                    logging.info('\t\tDefined features: ' + features)
                    self.features = features

        if 'syntax_constituency_tree' in features.keys():
            if features['syntax_constituency_tree'] == 'default':
                self.features['syntax_constituency_tree'] = default_features['syntax_constituency_tree']  # Defaults ziehen!



    def parse_sentence(self, sent_constituents):
        """Computes the Constituency properties of one sentence.

        Parameters
        ----------
        sent_constituents : complex

        Returns
        -------
            a tuple of s_non_terminals, s_doc_out_degrees, s_ntwl, s_graph, s_out_degrees
        """

        non_terminals = []
        edges = []
        ntwl = []
        out_degrees = []

        for const in sent_constituents:
            if len(list(const._.children)) > 0 or len(const._.labels) > 0:
                for lab in const._.labels:
                    non_terminals.append(lab)
                if len(const._.labels) > 1:  # uneray chain
                    i = 0
                    while i < len(list(const._.labels)):
                        out_degrees.append(1)
                        i += 1
                else:  # no unary chain
                    out_degrees.append(len(list(const._.children)))
                for lab in const._.labels:
                    ntwl.append(lab)
            for child in const._.children:
                edges.append(('{0}'.format(const), '{0}'.format(child)))

        return non_terminals, [value for value in out_degrees if value > 0], ntwl, nx.Graph(edges), out_degrees

    def feat_doc(self, doc, plain_text):
        """Get constituency metrics for a document

        Parameters
        ----------
        plain_text : basestring
        doc : spaCy Doc

        Returns
        -------
        dict
            dictionary with feature name of subfeatures as key and list of entities as value
        """

        tree_heights = []
        clauses = []

        non_terminals_wo_leaves = []
        non_terminals = []
        doc_constituents = []
        doc_parents = []

        t_units = []
        closeness_centrality = []
        out_degree_centralization = []
        doc_out_degrees = []

        try:
            doc_sents = self.const_nlp(plain_text)
            n_sents = len(list(doc_sents.sents))

            for sent in doc_sents:
                try:
                    parse_tree = ParentedTree.fromstring(sent._.parse_string)
                    tree_heights.append(parse_tree.height())
                except:
                    parse_tree = ParentedTree.fromstring('(' + sent._.parse_string + ')')
                    try:
                        tree_heights.append(parse_tree.height() - 1)
                    except:
                        logging.info('The parse string is not computable.')
                        sys.exit()

                s_non_terminals, s_doc_out_degrees, s_ntwl, s_graph, s_out_degrees = self.parse_sentence(sent._.constituents)

                non_terminals += s_non_terminals
                clauses += [len(const) for const in sent._.constituents if 'S' in const._.labels]

                doc_constituents += [const for const in sent._.constituents]
                doc_parents += [const for const in sent._.constituents if const._.parent is not None]
                doc_out_degrees += s_doc_out_degrees
                non_terminals_wo_leaves.append(len(s_ntwl))
                t_units += [len(const) for const in sent._.constituents if ('S' in const._.labels) and (const._.parent != None)]

                if s_out_degrees:
                    max_out_degree = np.max(s_out_degrees)
                else:
                    max_out_degree = 0

                div = len(s_graph) ** 2 - 2 * len(s_graph) + 1
                if div != 0:
                    out_degree_centralization.append(sum(max_out_degree - d for d in s_out_degrees) / div)
                else:
                    out_degree_centralization.append(0)
                if len(s_graph) > 1:
                    closeness_centrality += nx.algorithms.centrality.closeness_centrality(s_graph).values()
                else:
                    closeness_centrality += [0]

        except:

            n_sents = len(list(doc.sents))

            for sent in doc.sents:
                try:
                    doc_const = self.const_nlp(sent.text)

                except:
                    try:
                        doc_const = self.const_nlp(sent.text.replace('\n', ' '))
                    except:
                        try:
                            doc_const = self.const_nlp(sent.text.replace('\n', ' ').replace('[REF], ', ''))
                        except:
                            logging.info(sent.text)
                            raise ValueError('The sentences below is not computable.')

                for sent in doc_const.sents:
                    try:
                        parse_tree = ParentedTree.fromstring(sent._.parse_string)
                        tree_heights.append(parse_tree.height())
                    except:
                        parse_tree = ParentedTree.fromstring('(' + sent._.parse_string + ')')
                        try:
                            tree_heights.append(parse_tree.height() - 1)
                        except:
                            logging.info(sent.text)
                            raise ValueError('The parse string is not computable.')

                    s_non_terminals, s_doc_out_degrees, s_ntwl, s_graph, s_out_degrees = self.parse_sentence(sent._.constituents)

                    non_terminals += s_non_terminals
                    clauses += [len(const) for const in sent._.constituents if 'S' in const._.labels]
                    doc_constituents += [const for const in sent._.constituents]
                    doc_parents += [const for const in sent._.constituents if const._.parent is not None]
                    doc_out_degrees += s_doc_out_degrees
                    non_terminals_wo_leaves.append(len(s_ntwl))
                    t_units += [len(const) for const in sent._.constituents if ('S' in const._.labels) and (const._.parent != None)]

                    if s_out_degrees:
                        max_out_degree = np.max(s_out_degrees)
                    else:
                        max_out_degree = 0

                    div = len(s_graph) ** 2 - 2 * len(s_graph) + 1
                    if div != 0:
                        out_degree_centralization.append(sum(max_out_degree - d for d in s_out_degrees) / div)
                    else:
                        out_degree_centralization.append(0)
                    if len(s_graph) > 1:
                        closeness_centrality += nx.algorithms.centrality.closeness_centrality(s_graph).values()
                    else:
                        closeness_centrality += [0]

        if doc_out_degrees:
            m_doc_out_degrees = np.max(doc_out_degrees)
        else:
            m_doc_out_degrees = 0

        data = {}
        metrics = {}

        if tree_heights:
            metrics['AvgMaxDepth'] = np.mean(tree_heights)
        else:
            metrics['AvgMaxDepth'] = 0

        if doc_out_degrees:
            metrics['AvgFan'] = np.mean(doc_out_degrees)
        else:
            metrics['AvgFan'] = 0

        metrics['MaxFan'] = m_doc_out_degrees

        if non_terminals_wo_leaves:
            metrics['AvgNonTerminales_sent'] = np.mean(non_terminals_wo_leaves)
        else:
            metrics['AvgNonTerminales_sent'] = 0

        if n_sents > 0:
            metrics['AvgConstituents_sent'] = len(doc_constituents) / n_sents
            metrics['AvgTunits_sent'] = len(t_units) / n_sents
        else:
            metrics['AvgConstituents_sent'] = 0
            metrics['AvgTunits_sent'] = 0

        if len(doc_constituents) > 0:
            metrics['AvgLenConstituents'] = np.mean([len(i) for i in doc_constituents])
        else:
            metrics['AvgLenConstituents'] = 0

        len_t_units = len([i for i in t_units if i > 0])
        if len_t_units > 0:
            metrics['AvgLenTunits'] = sum(t_units) / len_t_units
        else:
            metrics['AvgLenTunits'] = 0

        if out_degree_centralization:
            metrics['AvgOutdegreeCentralization'] = np.mean(out_degree_centralization)
        else:
            metrics['AvgOutdegreeCentralization'] = 0

        if closeness_centrality:
            metrics['AvgClosenessCentralization'] = np.mean(closeness_centrality)
        else:
            metrics['AvgClosenessCentralization'] = 0

        if 'syntax_constituency_metrics' in self.features.keys():
            data['syntax_constituency_metrics'] = {'features': {f: metrics[f] for f in self.features['syntax_constituency_metrics']}}

            data['syntax_constituency_metrics']['syntax_const'] = {}
            data['syntax_constituency_metrics']['syntax_const']['tree_heights'] = tree_heights
            data['syntax_constituency_metrics']['syntax_const']['doc_out_degrees'] = doc_out_degrees
            data['syntax_constituency_metrics']['syntax_const']['non_terminals_wo_leaves'] = non_terminals_wo_leaves
            data['syntax_constituency_metrics']['syntax_const']['doc_const'] = doc_const
            data['syntax_constituency_metrics']['syntax_const']['t_units'] = t_units
            data['syntax_constituency_metrics']['syntax_const']['out_degree_centralization'] = out_degree_centralization
            data['syntax_constituency_metrics']['syntax_const']['closeness_centrality'] = closeness_centrality

        if 'syntax_constituency_tree' in self.features.keys():
            cnt = collections.Counter([item for item in non_terminals])
            data['syntax_constituency_tree'] = {}
            if doc._.n_tokens > 0:
                data['syntax_constituency_tree']['counts'] = dict(zip(cnt.keys(), [cnt[c] for c in cnt]))
            else:
                data['syntax_constituency_tree']['counts'] = dict(zip(cnt.keys(), [0 for _ in cnt]))

        return data

    def feat_corpus(self, corpus):

        """Get constituency metrics for a corpus

        Parameters
        ----------
        corpus : corpus

        Returns
        -------
        dict
            dictionary with subfeatures as key and syntax_const as key with interim results
        """

        data = {'features': {}}

        if corpus.resources.syntax_const['tree_heights']:
            data['features']['AvgMaxDepth'] = np.mean(corpus.resources.syntax_const['tree_heights'])
        else:
            data['features']['AvgMaxDepth'] = 0

        if corpus.resources.syntax_const['doc_out_degrees']:
            data['features']['AvgFan'] = np.mean(corpus.resources.syntax_const['doc_out_degrees'])
        else:
            data['features']['AvgFan'] = 0

        if corpus.resources.syntax_const['non_terminals_wo_leaves']:
            data['features']['AvgNonTerminales_sent'] = np.mean(corpus.resources.syntax_const['non_terminals_wo_leaves'])
        else:
            data['features']['AvgNonTerminales_sent'] = 0

        if corpus.resources.syntax_const['doc_const'] and corpus.sizes.sentences_cnt > 0:
            data['features']['AvgConstituents_sent'] = len(corpus.resources.syntax_const['doc_const']) / corpus.sizes.sentences_cnt
        else:
            data['features']['AvgConstituents_sent'] = 0

        if corpus.resources.syntax_const['t_units'] and corpus.sizes.sentences_cnt > 0:
            data['features']['AvgTunits_sent'] = len(corpus.resources.syntax_const['t_units']) / corpus.sizes.sentences_cnt
        else:
            data['features']['AvgTunits_sent'] = 0

        len_t_units = len([i for i in corpus.resources.syntax_const['t_units'] if i > 0])
        if len_t_units > 0:
            data['features']['AvgLenTunits'] = sum(corpus.resources.syntax_const['t_units']) / len_t_units
        else:
            data['features']['AvgLenTunits'] = 0

        if corpus.resources.syntax_const['out_degree_centralization']:
            data['features']['AvgOutdegreeCentralization'] = np.mean(corpus.resources.syntax_const['out_degree_centralization'])
        else:
            data['features']['AvgOutdegreeCentralization'] = 0

        if corpus.resources.syntax_const['closeness_centrality']:
            data['features']['AvgClosenessCentralization'] = np.mean(corpus.resources.syntax_const['closeness_centrality'])
        else:
            data['features']['AvgClosenessCentralization'] = 0

        return data

def init_syntax_constituency_metrics():
    return {
        'tree_heights': [],
        'doc_out_degrees': [],
        'non_terminals_wo_leaves': [],
        'doc_const': [],
        't_units': [],
        'out_degree_centralization': [],
        'closeness_centrality': []
    }

def update_syntax_constituency_metrics(syntax_const, data):

    for key in data:
        syntax_const[key] += data[key]

    return syntax_const
