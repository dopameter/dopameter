import logging
import pandas as pd
import os
import glob

from dopameter.configuration.installation import ConfLanguages


def run_aggregation(config):
    if os.path.isdir(config['output']['path_features']):

        logging.info('===============================================================================================')
        logging.info('Start clustering')
        logging.info('path_features: ' + config['output']['path_features'])
        logging.info('path_clusters: ' + config['output']['path_clusters'])
        logging.info('===============================================================================================')

        cluster = ClusterCorpora(
            corpora=config['corpora'],
            features=config['features'],
            path_features=config['output']['path_features'],
            path_clusters=config['output']['path_clusters'],
            feature_file_format_for_clustering=config['settings']['file_format_clustering'],
            diagram_file_formats=config['settings']['file_format_plots'],
            settings=config['cluster'],
            tasks=config['settings']['tasks'],
            file_format_features=config['settings']['file_format_features']
        )
        cluster.compute_clusters()
    else:
        exit("The given directory of features " + config['output']['path_features'] + " is not existing. No clusting started!")


class ClusterCorpora:
    """Aggregation: cluster corpora

    Parameters
    ----------

    corpora : dict
    features : dict
    path_features : str
    path_clusters : str
    feature_file_format_for_clustering : str
    diagram_file_formats : list of str or Iterable of str
    settings : dict
    tasks : list of str or Iterable of str
    file_format_features : list of str or Iterable of str

    Notes
    -----
    The analytics mode is able to compute dependencies between different corpora via 2 clustering modes:
        * [_k_-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans),
        * [_t_-distributed Stochastic Neighbor Embedding (t-SNE)](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE) with [DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan).
    Fot these steps, [scikit-learn](https://scikit-learn.org) library is used.
    This modular architecture is open to extension by a wider range of additional clustering algorithms and other machine learning libraries.

    * Configuration
        * `"corpora""`: Define the corpora for your analysis of clusters!
        * `"features"`: Define the feature sets for your analysis of clusters!

    * **Decentralized approach:**
      * You can run the task `"cluster"` on different places.
      * You can combine the features via task `"cluster"` if you copy the features `"path_features"` into one directory `"path_features"` and define it into one [configuration file](../res/example_configurations).

    * Name the 2 modes via in the configuration file, here default is activated:

    * `"cluster" : {
        "k-means": "default",
        "t-sne": "default"
      }`

    * _k_-means

        * Note: _k_-means used an internal [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler) of the features before clustering.
        * Default adjustments of _k_-means (following scores loaded automatically)
          * `"n_clusters":` : if you do not define the number of clusters, n is defined by the amount of corpora
          * `"random_state": "None"`
          * `"n_init": 50`
          * `"max_iter": 300`

        * Detailed configuration

              "features": {
                "lexical_diversity": "default",
                "ner": "default",
                "pos": "default",
                "emotion": "default",
                "token_characteristics": "default",
                "surface": "default",
                "syntax_dependency_metrics" : "default",
                "syntax_dependency_tree": "default"
              },
              "cluster" : {
                "k-means": {
                  "n_clusters": 5,
                  "random_state": "None",
                  "n_init": 50,
                  "max_iter": 300
                }
              }

    * _t_-SNE and DBSCAN

        * Default adjustments of t-sne (following scores loaded automatically)
          * `"n_components": 2`
          * `"random_state": 1`
          * `"perplexity": 500` - - Note: be aware, that the defined value of perplexity is at least the amount of single documents.
          * `"learning_rate": 500`
          * `"init": "pca"`
          * `"eps": 2`
          * `"min_samples": 5`

        * Detailed configuration

              "features": {
                "lexical_diversity": "default",
                "ner": "default",
                "pos": "default",
                "emotion": "default",
                "token_characteristics": "default",
                "surface": "default",
                "syntax_dependency_metrics" : "default",
                "syntax_dependency_tree": "default"
              },
              "cluster" : {
                "t-sne": {
                  "n_components": 2,
                  "random_state": 1,
                  "perplexity": 500,
                  "learning_rate": 500,
                  "init": "pca",
                  "eps": 2,
                  "min_samples": 5
                }
              }

    * Output
        * For every setup, there will be produced several pictures and cluster maps.
        * For every feature, a single directory is produced, including:
          * One figure with clustered data points of the corpora
          * For every cluster a figure with the highlighted cluster in the map of corpora
          * One figure with clustered data points of the clusters
          * For every corpus a figure with the highlighted cluster in the map of corpora
          * Cluster maps in table formatted files

    """

    def __init__(
        self,
        corpora,
        features,
        path_features,
        path_clusters,
        feature_file_format_for_clustering,
        diagram_file_formats,
        settings,
        tasks,
        file_format_features
    ):
        self.corpora = corpora
        self.features = features
        self.path_features = path_features
        self.feature_file_format_for_clustering = feature_file_format_for_clustering
        self.diagram_file_formats = diagram_file_formats
        self.tasks = tasks
        self.file_format_features = file_format_features

        if not os.path.isdir(path_clusters):
            os.mkdir(path_clusters)
        self.path_clusters = path_clusters

        self.settings = settings

        self.sns_rs = {
                    'figure.figsize': (10, 10),
                    'figure.facecolor': 'white',
                    'axes.facecolor': 'white',
                    'axes.edgecolor': 'black',
                    'axes.grid': False,
                    'axes.axisbelow': 'line',
                    'axes.labelcolor': 'black',
                    'grid.color': '#b0b0b0',
                    'grid.linestyle': '-',
                    'text.color': 'black',
                    'xtick.color': 'black',
                    'ytick.color': 'black',
                    'xtick.direction': 'out',
                    'ytick.direction': 'out',
                    'patch.edgecolor': 'black',
                    'patch.force_edgecolor': False,
                    'image.cmap': 'viridis',
                    'font.family': ['sans-serif'],
                    'font.sans-serif': ['DejaVu Sans',
                                        'Bitstream Vera Sans',
                                        'Computer Modern Sans Serif',
                                        'Lucida Grande',
                                        'Verdana',
                                        'Geneva',
                                        'Lucid',
                                        'Arial',
                                        'Helvetica',
                                        'Avant Garde',
                                        'sans-serif'],
                    'xtick.bottom': True,
                    'xtick.top': False,
                    'ytick.left': True,
                    'ytick.right': False,
                    'axes.spines.left': True,
                    'axes.spines.bottom': True,
                    'axes.spines.right': True,
                    'axes.spines.top': True
                }


        if 'k-means' in self.settings.keys():
            self.path_kmeans = self.path_clusters + os.sep + 'k-means'
            if not os.path.isdir(self.path_kmeans):
                os.mkdir(self.path_kmeans)

        if 't-sne' in self.settings.keys():
            self.path_tsne = self.path_clusters + os.sep + 't-sne'
            if not os.path.isdir(self.path_tsne):
                os.mkdir(self.path_tsne)


    def cluster_corpora(self, path_features):
        """prepare datasets of a given path of features"""

        feat_name = os.path.basename(path_features)
        feat_corpora = glob.glob(path_features + os.sep + '*.' + self.feature_file_format_for_clustering)
        dataset = pd.DataFrame()

        for f in feat_corpora:
            corpus = os.path.basename(f).replace('.' + self.feature_file_format_for_clustering, '').replace('_' + feat_name, '')

            if corpus in self.corpora:

                if self.feature_file_format_for_clustering == 'csv':
                    data_f = pd.read_csv(f)
                elif self.feature_file_format_for_clustering == 'excel' or self.feature_file_format_for_clustering == 'xlsx':
                    data_f = pd.read_excel(f)
                else:
                    raise ValueError(
                        'Your defined feature_file_format_for_clustering ' + self.feature_file_format_for_clustering + ' is not allowed. Allowed formats: csv, excel, xlsx!'
                    )

                if 'collection' in self.corpora[corpus].keys():
                    data_f['corpus'] = '[' + self.corpora[corpus]['collection'] + '] ' + corpus
                    data_f['collection'] = self.corpora[corpus]['collection']
                else:
                    data_f['collection'] = 'None'
                    data_f['corpus'] = corpus

                data_f['language'] = ConfLanguages().lang_def[self.corpora[corpus]['language']]

                dataset = pd.concat([dataset, data_f])
                logging.info('Loaded: ' + f)

        if 'k-means' in self.settings.keys():
            from dopameter.analytics.aggregation.kmeans import ClusterKMEANS
            c_kmeans = ClusterKMEANS(
                corpora=self.corpora,
                features=self.features,
                path_features=self.path_features,
                path_clusters=self.path_clusters,
                feature_file_format_for_clustering=self.feature_file_format_for_clustering,
                diagram_file_formats=self.diagram_file_formats,
                settings=self.settings,
                tasks=self.tasks,
                file_format_features=self.file_format_features
            )
            c_kmeans.cluster_kmeans_by_feature(dataset=dataset, feat_name=feat_name)

        if 't-sne' in self.settings.keys():
            from dopameter.analytics.aggregation.tsne import ClusterTSNE
            c_tsne = ClusterTSNE(
                corpora=self.corpora,
                features=self.features,
                path_features=self.path_features,
                path_clusters=self.path_clusters,
                feature_file_format_for_clustering=self.feature_file_format_for_clustering,
                diagram_file_formats=self.diagram_file_formats,
                settings=self.settings,
                tasks=self.tasks,
                file_format_features=self.file_format_features
            )
            c_tsne.cluster_tsne_by_feature(dataset=dataset, feat_name=feat_name)

        return dataset, feat_corpora

    def compute_clusters(self):
        """prepare configuration of a given configuration"""

        features = {os.path.basename(f): f.path for f in os.scandir(self.path_features) if f.is_dir()}

        if features:
            for feat in features:
                if feat in self.features.keys() and feat != 'ngrams':
                    self.cluster_corpora(features[feat])
                if '_ngrams_tfidf' in feat or '_ngrams_tfidf_pos' in feat:
                    n = int(feat.split('_')[0])
                    if 'ngrams' in self.features and n in self.features['ngrams']:
                        self.cluster_corpora(features[feat])
        else:
            exit("The given directory of features " + self.path_features + " is empty! No clusting started!")
