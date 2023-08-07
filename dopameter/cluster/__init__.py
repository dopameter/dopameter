import logging
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import seaborn as sns
import os
import glob

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from dopameter.output import write_df_to_file


def get_colors(n):
    """get colors for scatter plots by given number colors"""
    return [hsv_to_rgb(((1/n) * (i+0.5), 0.75, 0.75)) for i in range(n)]


class ClusterCorpora:
    """Aggregation: cluster corpora

    Parameters
    ----------

    corpora : dict
    features : dict
    path_features : str
    path_clusters : str
    feature_file_format_for_clustering : list of str
    diagram_file_formats : list of str
    settings : dict
    tasks : list of str
    file_format_features : list of str

    Notes
    -----
    The aggregation mode is able to compute dependencies between different corpora via 2 clustering modes:
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
                "lexical_richness": "default",
                "negation": "default",
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
                "lexical_richness": "default",
                "negation": "default",
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

    def cluster_tsne_by_feature(self, dataset, feat_name):
        """creates cluster of feature sets of t-sne and DBSCAN"""
        logging.info('================================================================================')
        logging.info('T-SNE Visualisation')
        logging.info('Feature ' + feat_name)
        logging.info('================================================================================')

        dataset = dataset.fillna(0).rename(columns={'Unnamed: 0': 'document'})
        logging.info('Dataset loaded.')

        path_tsne_feat = self.path_tsne + os.sep + feat_name
        if not os.path.isdir(path_tsne_feat):
            os.mkdir(path_tsne_feat)

        if self.settings['t-sne'] == 'default':
            self.settings['t-sne'] = {}
            self.settings['t-sne']['n_components'] = 2
            self.settings['t-sne']['random_state'] = 1
            self.settings['t-sne']['perplexity'] = 100
            self.settings['t-sne']['learning_rate'] = 500
            self.settings['t-sne']['init'] = 'pca',
            self.settings['t-sne']['eps'] = 2, # maximum distance between two samples
            self.settings['t-sne']['min_samples'] = 10, # number of samples (or total weight) in a neighborhood


        #if len(dataset) < int(self.settings['t-sne']['perplexity']):
        #    logging.warning('The length of your dataset is smaller than you configured perplexity: ' + len(dataset) + ' < ' + self.settings['t-sne']['perplexity'])
        #    logging.warning('The configured perplexity is by length of your dataset!')
        #    self.settings['t-sne']['perplexity'] = len(dataset)

        logging.info("Compute T-SNE scores with input values " + str(self.settings['t-sne']))

        tsne = TSNE(
            n_components=   int(self.settings['t-sne']['n_components']),
            random_state=   int(self.settings['t-sne']['random_state']),
            perplexity=     self.settings['t-sne']['perplexity'],
            learning_rate=  self.settings['t-sne']['learning_rate'],
            init=           self.settings['t-sne']['init']
        )

        x_data = tsne.fit_transform(
            X=dataset.drop(['document', 'corpus'], axis=1)
        )

        db = DBSCAN(
            eps=self.settings['t-sne']['eps'],
            min_samples=self.settings['t-sne']['min_samples']
        )

        get_clusters = db.fit_predict(x_data)
        labels = db.fit(x_data).labels_

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        logging.info("Estimated number of clusters: %d" % n_clusters_)
        logging.info("Estimated number of noise points: %d" % n_noise_)

        if 'plot' in self.tasks:

            sns.set(rc=self.sns_rs)

            colors = get_colors(n=len(set(labels)))
            dataset['x_data'] = x_data[:, 0]
            dataset['y_data'] = x_data[:, 1]

            for i, el in enumerate(set(labels)):

                sns.set(rc=self.sns_rs)
                sns.scatterplot(
                    x=x_data[:, 0],
                    y=x_data[:, 1],
                    hue=dataset['corpus'],
                    alpha=0.9,
                    marker='o'
                ).set(xlabel='x_data', ylabel='y_data')

                plt.title(label=feat_name + ' T-SNE cluster - all corpora')
                plt.legend(
                    bbox_to_anchor=(1.02, 1),
                    loc='upper left',
                    borderaxespad=0
                )
                plt.savefig(
                    path_tsne_feat + os.sep + feat_name + '_all_corpora_t-sne.png',
                    bbox_inches='tight',
                    format='png'
                )

                if el != -1:
                    l = 'Cluster ' + str(el)
                    marker = '2'
                else:
                    l = 'Noise'
                    marker = '^'

                plt.scatter(
                    x_data[get_clusters == el, 0],
                    x_data[get_clusters == el, 1],
                    label=l,
                    color = 'black',
                    alpha=0.9,
                    marker=marker
                )
                plt.legend(
                    bbox_to_anchor=(1.02, 1),
                    loc='upper left',
                    borderaxespad=0
                )
                plt.title(label=feat_name + ' T-SNE cluster - all corpora - ' + l)
                plt.savefig(
                    path_tsne_feat + os.sep + feat_name + '_all_corpora_t-sne_' + l.replace(' ', '_') + '.png',
                    bbox_inches='tight',
                    format='png'
                )
                plt.close()
                plt.clf()

            for c in self.corpora.keys():

                for i, el in enumerate(set(labels)):

                    if el != -1:
                        l = 'Cluster ' + str(el)
                        marker = 'o'
                    else:
                        l = 'Noise'
                        marker = '<'

                    plt.scatter(
                        x_data[get_clusters == el, 0],
                        x_data[get_clusters == el, 1],
                        label=l,
                        color=colors[i],
                        alpha=0.9,
                        marker=marker
                    )
                    plt.xlabel('x_data')
                    plt.ylabel('y_data')

                plt.legend(
                    bbox_to_anchor=(1.02, 1),
                    loc='upper left',
                    borderaxespad=0
                )
                plt.title(label=feat_name + ' - T-SNE cluster')

                plt.savefig(
                    path_tsne_feat + os.sep + feat_name + '_t-sne.png',
                    bbox_inches='tight',
                    format='png'
                )
                logging.info('Plot: ' + path_tsne_feat + os.sep + feat_name + '_t-sne.png')

                sns.set(rc=self.sns_rs)

                y = dataset[dataset['corpus']==c]['corpus']

                sns.scatterplot(
                    x=dataset[dataset['corpus']==c]['x_data'],
                    y=dataset[dataset['corpus']==c]['y_data'],
                    hue=y,
                    palette=['black'],
                    marker="2",
                    s=25
                ).set(title='t-SNE ' + feat_name, xlabel='x_data', ylabel='y_data')

                plt.legend(
                    bbox_to_anchor=(1.02, 1),
                    loc='upper left',
                    borderaxespad=0
                )

                plt.savefig(
                    path_tsne_feat + os.sep + feat_name + '_tsne_' + c + '.png',
                    bbox_inches='tight',
                    format='png'
                )
                plt.close()
                plt.clf()
                logging.info('Plot: ' + path_tsne_feat + os.sep + feat_name + '_tsne_' + c + '.png')

        tsne_frame = pd.DataFrame(x_data)
        tsne_frame['document'] = dataset['document'].values
        tsne_frame['corpus'] = dataset['corpus'].values
        tsne_frame['clusters'] = get_clusters

        write_df_to_file(
            data=tsne_frame,
            path_file=path_tsne_feat + os.sep + feat_name + '_tsne_dataset',
            file_format_features=self.file_format_features
        )

        logging.info('-----------------')


    def cluster_kmeans_by_feature(self, dataset, feat_name):
        """creates cluster of feature sets of k-means"""

        logging.info('================================================================================')
        logging.info('K-Means Visualisation')
        logging.info('Feature ' + feat_name)
        logging.info('================================================================================')


        if self.settings['k-means'] == 'default':
            self.settings['k-means'] = {}
            self.settings['k-means']['n_clusters'] = len(self.corpora)
            self.settings['k-means']['random_state'] = None
            self.settings['k-means']['n_init'] = 'auto'
            self.settings['k-means']['max_iter'] = 300
        else:
            if self.settings['k-means']['random_state'] == 'None':
                self.settings['k-means']['random_state'] = None

        dataset = dataset.fillna(0).rename(columns={'Unnamed: 0': 'document'})
        logging.info('Dataset loaded.')

        path_kmeans_feat = self.path_kmeans + os.sep + feat_name
        if not os.path.isdir(path_kmeans_feat):
            os.mkdir(path_kmeans_feat)

        y = dataset['corpus']
        le = LabelEncoder()
        label = le.fit_transform(y)

        n_clusters = self.settings['k-means']['n_clusters']

        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()

        x_scaled = min_max_scaler.fit_transform(dataset.drop(['document', 'corpus'], axis=1))
        pd_x_scaled = pd.DataFrame(x_scaled, columns=dataset.drop(['document', 'corpus'], axis=1).columns)

        kmeans = KMeans(
            n_clusters=self.settings['k-means']['n_clusters'],#n_clusters,
            random_state=self.settings['k-means']['random_state'],#None,#1,
            n_init=self.settings['k-means']['n_init'],#'auto',# default 10  #20,
            max_iter=self.settings['k-means']['max_iter'],#300 # default 300
        )

        y_data = kmeans.fit_transform(pd_x_scaled, label)
        y_pred = kmeans.predict(pd_x_scaled)

        colors = get_colors(n=n_clusters)

        for i in range(n_clusters):

            sns.set(rc={'figure.figsize': (10, 10)})
            sns.scatterplot(
                x=y_data[:, 0],
                y=y_data[:, 1],
                hue=y, # equal to dataset['corpus']
                s=25
            ).set(title='K-Means ' + feat_name)

            plt.legend(
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                borderaxespad=0
            )

            plt.savefig(path_kmeans_feat + os.sep + feat_name + '_all_corpora_kmeans.png', bbox_inches='tight', format='png')
            logging.info('Plot: ' + path_kmeans_feat + os.sep + feat_name + '_all_corpora_kmeans.png')

            sns.set_style(style='white')
            sns.set(rc={'figure.figsize': (10, 10)})

            plt.scatter(
                y_data[y_pred == i, 0],
                y_data[y_pred == i, 1],
                label='Cluster ' + str(i+1),
                color = 'black',
                alpha=0.5,
                marker='2'
            )

            plt.legend(
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                borderaxespad=0
            )
            plt.title(feat_name + ' - K-Means cluster')
            plt.savefig(path_kmeans_feat + os.sep + feat_name + '_all_corpora_kmeans_Cluster_' + str(i) + '.png', bbox_inches='tight', format='png')
            logging.info('Plot: ' + path_kmeans_feat + os.sep + feat_name + '_all_corpora_kmeans_Cluster_' + str(i) + '.png')
            plt.close()

        for c in self.corpora.keys():

            sns.set(rc={'figure.figsize': (10, 10)})
            for i in range(n_clusters):
                plt.scatter(
                    y_data[y_pred == i, 0],
                    y_data[y_pred == i, 1],
                    label='Cluster ' + str(i + 1),
                    color = colors[i],
                    alpha=0.5,
                )

                plt.legend(
                    bbox_to_anchor=(1.02, 1),
                    loc='upper left',
                    borderaxespad=0
                )
            plt.title(feat_name + ' - K-Means cluster - 3')
            plt.savefig(path_kmeans_feat + os.sep + feat_name + '_kmeans.png', bbox_inches='tight', format='png')
            logging.info('Plot: ' + path_kmeans_feat + os.sep + feat_name + '_kmeans.png')

            dataset['x_data'] = y_data[:, 0]
            dataset['y_data'] = y_data[:, 1]

            sns.scatterplot(
                x=dataset[dataset['corpus'] == c]['x_data'],  # x_data[:, 0],
                y=dataset[dataset['corpus'] == c]['y_data'],  # x_data[:, 1],
                hue=dataset[dataset['corpus'] == c]['corpus'],
                palette=['black'],
                marker="2",
                s=25
            ).set(title='K-Means ' + feat_name)

            plt.legend(
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                borderaxespad=0
            )

            plt.savefig(
                path_kmeans_feat + os.sep + feat_name + '_kmeans_' + c + '.png',
                bbox_inches='tight',
                format='png'
            )
            plt.close()
            plt.clf()
            logging.info('Plot: ' + path_kmeans_feat + os.sep + feat_name + '_kmeans_' + c + '.png')

        dataset['cluster'] = kmeans.labels_
        write_df_to_file(dataset, path_kmeans_feat + os.sep + feat_name + '_cluster_map', file_format_features='csv')

        logging.info('-----------------')


    def cluster_corpora(self, path_features):
        """prepare datasets of a given path of features"""

        feat_name = os.path.basename(path_features)
        feat_corpora = glob.glob(path_features + os.sep + '*.' + self.feature_file_format_for_clustering)
        dataset = pd.DataFrame()

        for f in feat_corpora:
            corpus_name = os.path.basename(f).replace('.' + self.feature_file_format_for_clustering, '')
            corpus_name = corpus_name.replace('_' + feat_name, '')

            if corpus_name in self.corpora:

                if self.feature_file_format_for_clustering == 'csv':
                    data_f = pd.read_csv(f)
                elif self.feature_file_format_for_clustering == 'excel' or self.feature_file_format_for_clustering == 'xlsx':
                    data_f = pd.read_excel(f)
                else:
                    raise ValueError(
                        'Your defined feature_file_format_for_clustering ' + self.feature_file_format_for_clustering +
                        ' is wrong. Allowed formats are: csv, excel, xlsx!'
                    )

                data_f['corpus'] = corpus_name
                dataset = pd.concat([dataset, data_f])
                logging.info('Loaded: ' + f)

        if 'k-means' in self.settings.keys():
            self.cluster_kmeans_by_feature(dataset, feat_name)
        if 't-sne' in self.settings.keys():
            self.cluster_tsne_by_feature(dataset, feat_name)
        return dataset, feat_corpora

    def compute_clusters(self):
        """prepare configuration of a given configuration"""

        features = {os.path.basename(f): f.path for f in os.scandir(self.path_features) if f.is_dir()}

        if features:
            for feat in features:
                if feat in self.features.keys() and feat != 'ngrams':
                    self.cluster_corpora(features[feat])
                if '_ngrams_tfidf' in feat:
                    n = int(feat.split('_')[0])
                    if 'ngrams' in self.features and n in self.features['ngrams']:
                        self.cluster_corpora(features[feat])
        else:
            exit("The given directory of features " + self.path_features + " is empty! No clusting started!")
