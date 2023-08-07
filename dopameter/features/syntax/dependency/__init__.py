import collections
import numpy as np
import networkx as nx
import logging


class DependencyFeatures:
    """Get Syntax Dependency Metrics

    Parameters
    ----------
    features : list, default={
                        'syntax_dependency_metrics' : 'default',
                        'syntax_dependency_tree': 'default'
                        }

    Attributes
    ----------
    * This packes is derived into 2 parts:
        * 'syntax_dependency_metrics` : Metrics / Scores
        * 'syntax_dependency_tree` : occurrences of parse node trees
    * `syntax_dependency_metrics` :
        * `MaxFan` : the maximum fan-out of the parse tree, i.e., the highest number of directly dominated nodes from an
            ancestor node in the entire parse tree, is determined over all sentences, and the maximum of all sentences
            of the document is selected
        * `AvgFan` : the averaged sum of all MaxFan scores per document
        * `AvgMaxDepth` : the maximum depth for each parse tree, i.e., the longest path from the root node to a leaf node,
            is determined for each sentence, summed up for all sentences of a document and averaged over all sentences,
            to find the maximum depth (height) of the dependency tree by its root, code adapted from
            https://stackoverflow.com/questions/35920826/how-to-find-height-for-non-binary-tree
            and https://gist.github.com/drussellmrichie/47deb429350e2e99ffb3272ab6ab216a
        * `AvgDepDist` : average score of the linear distance between two syntactically related words in a sentence
        * `MaxDepDist* : maximum score of the linear distance between two syntactically related words in a sentence
        * `AvgOutdegreeCentralization` : avergage of outdegree centralization value of all sentences of document
            * computed over all dependency graphs over single sentences of the document)
            * value in {0,1};
            * if value == 1: all other nodes are dependent on root node and height is 1
            * computed by out-degrees values of all n nodes (count of child nodes per single node),
                maximal out-degree
        * `AvgClosenessCentralization` : avergage closeness centralization of all all dependency graphs of sentences of
        document;
            * closeness centralization is calculated as the reciprocal of the sum of the length of
            the distance (shortest paths) between two nodes x and y and x (d(x,y)), y in 1..n of a graph with n nodes
            (Bavelas 1950): (computed over all dependency graphs over single sentences of the document);
            * per default 0 if count of nodes is 1

    Notes
    -----

    define in configuration .json file under features:

    * default:
      "tasks": ["features", "counts"],
      "features": {
        "syntax_dependency_metrics" : "default",
        "syntax_dependency_tree": "default",
      }
    * or in detail:
        "tasks": ["features", "counts"],
        "features": {
        "syntax_dependency_metrics" : [
            'AvgFan',
            'MaxFan',
            'AvgMaxDepth',
            'AvgDepDist',
            'MaxDepDist',
            'AvgOutdegreeCentralization',
            'AvgClosenessCentralization'
            ],
    "syntax_constituency_tree": ["ROOT", "CAP", "CS", "VP"], # only three examples
  }
    """
    def __init__(
            self,
            features={
            'syntax_depenendy_metrics' : 'default',
            'syntax_dependency_tree': 'default'
        }
    ):
        default_features = {
            'syntax_dependency_metrics' : [
                'AvgFan',
                'MaxFan',
                'AvgMaxDepth',
                'AvgDepDist',
                'MaxDepDist',
                'AvgOutdegreeCentralization',
                'AvgClosenessCentralization'
            ],
            'syntax_dependency_tree': 'default'
        }
        logging.info('\tInitialize syntax dependency features.')

        self.features = {}

        if 'syntax_dependency_metrics' in features.keys():
            if features['syntax_dependency_metrics'] == 'default':
                self.features['syntax_dependency_metrics'] = default_features['syntax_dependency_metrics']
            else:
                if set(features['syntax_dependency_metrics']).intersection(default_features['syntax_dependency_metrics']) == set():
                    raise ValueError('Your syntax dependency ' + ' '.join(features['syntax_dependency_metrics']) + ' are not defined! Allowed definitions: ', default_features['syntax_dependency_metrics'])
                else:
                    logging.info('\t\tDefined features: ' + features)
                    self.features['syntax_dependency_metrics'] = features['syntax_dependency_metrics']

        if 'syntax_dependency_tree' in features.keys():
            if features['syntax_dependency_tree'] == 'default':
                self.features['syntax_dependency_tree'] = default_features['syntax_dependency_tree']  # Defaults ziehen!


    def _tree_height(self, root):
        """Get recursive computed dependency tree height value
        Parameters
        ----------
        root : spaCy Token object

        Returns
        -------
        int
            maximum height of sentence's dependency parse tree
        """
        if not list(root.children):
            return 1
        else:
            return 1 + max(self._tree_height(x) for x in root.children)

    def feat_doc(self, doc):
        """Get dependency metrics features for a document

        Parameters
        ----------
        doc : spaCy Doc

        Returns
        -------
        dict
            dictionary with feature name of subfeatures as key and list of entities as value
        """

        data = {}
        metrics = {}

        if 'syntax_dependency_metrics' in self.features.keys():

            children_per_node = [value for value in [len([child for child in token.children]) for token in doc] if value > 0]
            if children_per_node:
                metrics['AvgFan'] = np.mean(children_per_node)
                metrics['MaxFan'] = np.max(children_per_node)
            else:
                metrics['AvgFan'] = 0
                metrics['MaxFan'] = 0

            max_depth = [self._tree_height(root) for root in [sent.root for sent in doc.sents]]
            if max_depth:
                metrics['AvgMaxDepth'] = np.mean(max_depth)
            else:
                metrics['AvgMaxDepth'] = 0

            closeness_centrality = []
            out_degree_centralization = []
            for sent in doc.sents:
                edges = []
                for token in sent:
                    for child in token.children:
                        edges.append(('{0}'.format(token), '{0}'.format(child)))

                graph = nx.Graph(edges)

                out_degrees = [len([child for child in token.children]) for token in sent]
                max_out_degree = np.max(out_degrees)

                div = (len(graph) ** 2 - 2 * len(graph) + 1)

                if div != 0:
                    out_degree_centralization.append(sum(max_out_degree - d for d in out_degrees) / div)
                else:
                    out_degree_centralization.append(0)

                if len(graph) > 1:
                    closeness_centrality += nx.algorithms.centrality.closeness_centrality(graph).values()

                else:
                    closeness_centrality += [0]

            dep_distance = [sum([np.abs(token.head.i - token.i) / len(sent) for token in sent]) for sent in doc.sents]
            if not dep_distance:
                metrics['AvgDepDist'] = 0
                metrics['MaxDepDist'] = 0
            else:
                metrics['AvgDepDist'] = np.mean(dep_distance)
                metrics['MaxDepDist'] = np.max(dep_distance)

            metrics['AvgOutdegreeCentralization'] = np.mean(out_degree_centralization)
            metrics['AvgClosenessCentralization'] = np.mean(closeness_centrality)

            data['syntax_dependency_metrics'] = {'features': {f: metrics[f] for f in self.features['syntax_dependency_metrics']}}

            data['syntax_dependency_metrics']['syntax_dep'] = {}
            data['syntax_dependency_metrics']['syntax_dep']['children_per_node'] = children_per_node
            data['syntax_dependency_metrics']['syntax_dep']['max_depth'] = max_depth
            data['syntax_dependency_metrics']['syntax_dep']['dep_distance'] = dep_distance
            data['syntax_dependency_metrics']['syntax_dep']['out_degree_centralization'] = out_degree_centralization
            data['syntax_dependency_metrics']['syntax_dep']['closeness_centrality'] = closeness_centrality

        if 'syntax_dependency_tree' in self.features.keys():
            cnt = collections.Counter([str(token.dep_) for token in doc if token.dep_])
            data['syntax_dependency_tree'] = {}
            if doc._.n_tokens > 0:
                data['syntax_dependency_tree']['counts'] = dict(zip(cnt.keys(), [cnt[c] for c in cnt]))
            else:
                data['syntax_dependency_tree']['counts'] = dict(zip(cnt.keys(), [0 for _ in cnt]))

        return data

    def feat_corpus(self, corpus):

        """Get dependency metrics for a corpus

        Parameters
        ----------
        corpus : corpus

        Returns
        -------
        dict
            dictionary with subfeatures as key and syntax_dep as key with interim results
        """

        data = {'features': {}}

        if corpus.syntax_dep['children_per_node']:
            data['features']['AvgFan'] = np.mean(corpus.syntax_dep['children_per_node'])
            data['features']['MaxFan'] = np.max(corpus.syntax_dep['children_per_node'])
        else:
            data['features']['AvgFan'] = 0
            data['features']['MaxFan'] = 0

        if corpus.syntax_dep['max_depth']:
            data['features']['AvgMaxDepth'] = np.mean(corpus.syntax_dep['max_depth'])
        else:
            data['features']['AvgMaxDepth'] = 0

        if not corpus.syntax_dep['dep_distance']:
            data['features']['AvgDepDist'] = 0
            data['features']['MaxDepDist'] = 0
        else:
            data['features']['AvgDepDist'] = np.mean(corpus.syntax_dep['dep_distance'])
            data['features']['MaxDepDist'] = np.max(corpus.syntax_dep['dep_distance'])

        if corpus.syntax_dep['out_degree_centralization']:
            data['features']['AvgOutdegreeCentralization'] = np.mean(corpus.syntax_dep['out_degree_centralization'])
        else:
            data['features']['AvgOutdegreeCentralization'] = 0

        if corpus.syntax_dep['closeness_centrality']:
            data['features']['AvgClosenessCentralization'] = np.mean(corpus.syntax_dep['closeness_centrality'])
        else:
            data['features']['AvgClosenessCentralization'] = 0

        return data
