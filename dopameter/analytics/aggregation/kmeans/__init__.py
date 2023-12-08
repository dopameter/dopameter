import logging
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from dopameter.analytics.aggregation import ClusterCorpora
from dopameter.analytics.summarization import df_to_file
from dopameter.analytics.vis_utils import get_colors, overview_cluster_per_dataset

def create_plots(dataset, level_to_plot, le, kmeans, pd_x_scaled, n_clusters, feat_name, path_kmeans_feat):

    """ create plots for different level of composition of corpora, collection, languages - k-means based plotting

        Parameters
        ----------

        dataset : dataframe
        level_to_plot : array of str
        le : Any
        kmeans : Any
        pd_x_scaled : dataframe
        n_clusters : int
        feat_name = str
        path_kmeans_feat : str

        labels : Any
        feat_name : str
        path_tsne_feat : str
        get_clusters : Any
    """

    path_kmeans_feat = path_kmeans_feat + os.sep + level_to_plot
    if not os.path.isdir(path_kmeans_feat):
        os.mkdir(path_kmeans_feat)

    y = dataset[level_to_plot]
    label = le.fit_transform(y)

    y_data = kmeans.fit_transform(pd_x_scaled, label)
    y_pred = kmeans.predict(pd_x_scaled)

    colors = get_colors(n=n_clusters)

    for i in range(n_clusters):
        sns.set(rc={'figure.figsize': (10, 10)})
        sns.scatterplot(
            x=y_data[:, 0],
            y=y_data[:, 1],
            hue=y,  # equal to dataset['corpus']
            s=25
        ).set(title='K-Means ' + feat_name + ' [' + level_to_plot + ']')

        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            borderaxespad=0
        )

        plt.savefig(
            path_kmeans_feat + os.sep + feat_name + '_' + level_to_plot + ' _kmeans.png',
            bbox_inches='tight',
            format='png'
        )
        logging.info('Plot: ' + path_kmeans_feat + os.sep + feat_name + '_all_corpora_kmeans.png')

        sns.set_style(style='white')
        sns.set(rc={'figure.figsize': (10, 10)})

        plt.scatter(
            y_data[y_pred == i, 0],
            y_data[y_pred == i, 1],
            label='Cluster ' + str(i),
            color='black',
            alpha=0.5,
            marker='2'
        )

        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            borderaxespad=0
        )
        plt.title(label="K-Means - feature set '" + feat_name + "' [" + level_to_plot + ']')
        plt.savefig(path_kmeans_feat + os.sep + feat_name + '_all_corpora_kmeans_Cluster_' + str(i) + '.png', bbox_inches='tight', format='png')
        logging.info('Plot: ' + path_kmeans_feat + os.sep + feat_name + '_all_corpora_kmeans_Cluster_' + str(i) + '.png')
        plt.close()

    if level_to_plot == 'corpus':
        iterate_over = sorted(dataset['corpus'].unique())
    elif level_to_plot == 'collection':
        iterate_over = sorted(dataset['collection'].unique())
    elif level_to_plot == 'language':
        iterate_over = sorted(dataset['language'].unique())

    for c in iterate_over:

        sns.set(rc={'figure.figsize': (10, 10)})
        for i in range(n_clusters):
            plt.scatter(
                y_data[y_pred == i, 0],
                y_data[y_pred == i, 1],
                label='Cluster ' + str(i),
                color=colors[i],
                alpha=0.5,
            )

            plt.legend(
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                borderaxespad=0
            )
        plt.title(label="K-Means - feature set '" + feat_name + "' [" + level_to_plot + '] ')
        plt.savefig(path_kmeans_feat + os.sep + feat_name + '_kmeans.png', bbox_inches='tight', format='png')
        logging.info('Plot: ' + path_kmeans_feat + os.sep + feat_name + '_kmeans.png')

        dataset['x_data'] = y_data[:, 0]
        dataset['y_data'] = y_data[:, 1]

        sns.scatterplot(
            x=dataset[dataset[level_to_plot] == c]['x_data'],  # x_data[:, 0],
            y=dataset[dataset[level_to_plot] == c]['y_data'],  # x_data[:, 1],
            hue=dataset[dataset[level_to_plot] == c][level_to_plot],
            palette=['black'],
            marker="2",
            s=25
        ).set(title="K-Means - feature set '" + feat_name + "' [" + level_to_plot + '] ')

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


class ClusterKMEANS(ClusterCorpora):

    """Get Aggregation and Clustering by K-Means

    Parameters
    ----------
    corpora : dictionary,
    features : dictionary,
    path_features : str,
    path_clusters : str,
    feature_file_format_for_clustering : str,
    diagram_file_formats : list of str,
    settings : dictionary,
    tasks : list,
    file_format_features : list of str

    Notes
    -----
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

        super().__init__(
            corpora,
            features,
            path_features,
            path_clusters,
            feature_file_format_for_clustering,
            diagram_file_formats,
            settings,
            tasks,
            file_format_features
        )


    def cluster_kmeans_by_feature(self, dataset, feat_name):
        """creates cluster of feature sets of k-means

        Parameters
        ----------

        dataset : dataframe
        feat_name : str

        """

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

        n_clusters = self.settings['k-means']['n_clusters']

        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()

        x_scaled = min_max_scaler.fit_transform(dataset.drop(['document', 'corpus', 'collection', 'language'], axis=1))
        pd_x_scaled = pd.DataFrame(x_scaled, columns=dataset.drop(['document', 'corpus', 'collection', 'language'], axis=1).columns)

        kmeans = KMeans(
            n_clusters=self.settings['k-means']['n_clusters'],  # n_clusters,
            random_state=self.settings['k-means']['random_state'],  # None,#1,
            n_init=self.settings['k-means']['n_init'],  # 'auto',# default 10  #20,
            max_iter=self.settings['k-means']['max_iter'],  # 300 # default 300
        )
        le = LabelEncoder()

        if 'level' in self.settings.keys():
            level = self.settings['level']
        else:
            level = ['corpus', 'collection', 'language']

        if 'corpus' in level:
            create_plots(
                dataset=dataset,
                level_to_plot='corpus',
                le=le,
                kmeans=kmeans,
                pd_x_scaled=pd_x_scaled,
                n_clusters=n_clusters,
                feat_name=feat_name,
                path_kmeans_feat=path_kmeans_feat
            )

        if sorted(dataset['collection'].unique()) != ['None'] and 'collection' in level:
            create_plots(
                dataset=dataset,
                level_to_plot='collection',
                le=le,
                kmeans=kmeans,
                pd_x_scaled=pd_x_scaled,
                n_clusters=n_clusters,
                feat_name=feat_name,
                path_kmeans_feat=path_kmeans_feat
            )

        if 'language' in level:
            create_plots(
                dataset=dataset,
                level_to_plot='language',
                le=le,
                kmeans=kmeans,
                pd_x_scaled=pd_x_scaled,
                n_clusters=n_clusters,
                feat_name=feat_name,
                path_kmeans_feat=path_kmeans_feat
            )

        dataset['cluster'] = kmeans.labels_
        df_to_file(
            data=dataset,
            path_file=path_kmeans_feat + os.sep + feat_name + '_cluster_map',
            file_format_features='csv'
        )

        overview_cluster_per_dataset(
            data=dataset,
            feature=feat_name,
            path=path_kmeans_feat
        )

        logging.info('-----------------')
