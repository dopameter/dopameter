import logging
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

from dopameter.analytics.aggregation import ClusterCorpora
from dopameter.analytics.summarization import df_to_file
from dopameter.analytics.vis_utils import overview_cluster_per_dataset, clean_file_name


def create_plots_by_kmeans(
        dataset,
        level_to_plot,
        kmeans,
        pd_x_scaled,
        n_clusters,
        feat_name,
        path_kmeans_feat,
        n_items,
        file_format_plots
    ):

    """ create plots for different level of composition of corpora, collection, languages - k-means based plotting

        Parameters
        ----------

        dataset : dataframe
        level_to_plot : array of str
        kmeans : Any
        pd_x_scaled : dataframe
        n_clusters : int
        feat_name = str
        path_kmeans_feat : str

        labels : Any
        feat_name : str
        path_tsne_feat : str
        get_clusters : Any
        n_items : int
        file_format_plots : str
    """

    path_kmeans_feat = path_kmeans_feat + os.sep + level_to_plot
    if not os.path.isdir(path_kmeans_feat):
        os.mkdir(path_kmeans_feat)

    y = dataset[level_to_plot]
    y_data = kmeans.fit_transform(X=pd_x_scaled, y=LabelEncoder().fit_transform(y))
    y_pred = kmeans.predict(pd_x_scaled)

    n = n_items
    colors = (sns.color_palette("flare", int(n / 2)) + sns.color_palette("viridis", int(n / 2) + int(n % 2)))
    colors2 = (sns.color_palette("flare", int(n_clusters / 2)) + sns.color_palette("viridis", int(n_clusters / 2) + int(n_clusters % 2)))

    for i in range(n_clusters):
        sns.set(rc={'figure.figsize': (10, 10)})
        sns.scatterplot(
            x=y_data[:, 0],
            y=y_data[:, 1],
            hue=y,  # equal to dataset['corpus']
            s=4,
            palette=colors,
        ).set(title='K-Means ' + feat_name + ' [' + level_to_plot + ']')

        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            borderaxespad=0
        )

        for file_format in file_format_plots:
            plt.savefig(
                path_kmeans_feat + os.sep + clean_file_name(feat_name + '_' + level_to_plot + ' _kmeans.' + file_format),
                bbox_inches='tight',
                format=file_format
            )
            logging.info('Plot: ' + path_kmeans_feat + os.sep + feat_name + '_all_corpora_kmeans.' + file_format)

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

        for file_format in file_format_plots:
            plt.savefig(
                path_kmeans_feat + os.sep + clean_file_name(feat_name + '_all_corpora_kmeans_Cluster_' + str(i) + '.' + file_format),
                bbox_inches='tight',
                format=file_format
            )
            logging.info('Plot: ' + path_kmeans_feat + os.sep + feat_name + '_all_corpora_kmeans_Cluster_' + str(i) + '.' + file_format)
        plt.close()
        plt.clf()

    if level_to_plot == 'corpus':
        iterate_over = sorted(dataset['corpus'].unique())
    elif level_to_plot == 'collection':
        iterate_over = sorted(dataset['collection'].unique())
    elif level_to_plot == 'language':
        iterate_over = sorted(dataset['language'].unique())
    else:
        exit(0)

    for c in iterate_over:

        sns.set(rc={'figure.figsize': (10, 10)})
        for i in range(n_clusters):
            plt.scatter(
                y_data[y_pred == i, 0],
                y_data[y_pred == i, 1],
                label='Cluster ' + str(i),
                color=colors2[i],
                alpha=0.5,
            )

            plt.legend(
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                borderaxespad=0
            )
        plt.title(label="K-Means - feature set '" + feat_name + "' [" + level_to_plot + '] ')

        for file_format in file_format_plots:
            plt.savefig(clean_file_name(path_kmeans_feat + os.sep + feat_name + '_kmeans.' + file_format),
                bbox_inches='tight',
                format=file_format
            )
            logging.info('Plot: ' + path_kmeans_feat + os.sep + feat_name + '_kmeans.' + file_format)

        dataset['x_data'] = y_data[:, 0]
        dataset['y_data'] = y_data[:, 1]

        sns.scatterplot(
            x=dataset[dataset[level_to_plot] == c]['x_data'],  # x_data[:, 0],
            y=dataset[dataset[level_to_plot] == c]['y_data'],  # x_data[:, 1],
            hue=dataset[dataset[level_to_plot] == c][level_to_plot],
            palette=['black'],
            marker="2",
            s=4
        ).set(title="K-Means - feature set '" + feat_name + "' [" + level_to_plot + ']')

        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            borderaxespad=0
        )

        for file_format in file_format_plots:
            plt.savefig(
                path_kmeans_feat + os.sep + clean_file_name(feat_name + '_kmeans_' + c + '.' + file_format),
                bbox_inches='tight',
                format=file_format
            )
            logging.info('Plot: ' + path_kmeans_feat + os.sep + feat_name + '_kmeans_' + c + '.' + file_format)
        plt.close()
        plt.clf()



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
        self.path_kmeans = self.path_clusters + os.sep + 'k-means'
        if not os.path.isdir(self.path_kmeans):
            os.mkdir(self.path_kmeans)

    def cluster_kmeans_by_feature(self, dataset, feat_name, file_format_plots):
        """creates cluster of feature sets of k-means

        Parameters
        ----------

        dataset : dataframe
        feat_name : str
        file_format_plots : [str]

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

        logging.info('Load Dataset.')
        dataset = dataset.fillna(0).rename(columns={'Unnamed: 0': 'document'})

        logging.info('Dataset loaded.')

        path_kmeans_feat = self.path_kmeans + os.sep + feat_name
        if not os.path.isdir(path_kmeans_feat):
            os.mkdir(path_kmeans_feat)

        n_clusters = self.settings['k-means']['n_clusters']

        pd_x_scaled = pd.DataFrame(
            preprocessing.MinMaxScaler().fit_transform(dataset.drop(['document', 'corpus', 'collection', 'language'], axis=1)),
            columns=dataset.drop(['document', 'corpus', 'collection', 'language'], axis=1).columns
        )

        logging.info("Compute k-Means scores with input configuration " + str(self.settings['k-means']))

        kmeans = KMeans(
            n_clusters=self.settings['k-means']['n_clusters'],  # n_clusters,
            random_state=self.settings['k-means']['random_state'],  # None,#1,
            n_init=self.settings['k-means']['n_init'],  # 'auto',# default 10  #20,
            max_iter=self.settings['k-means']['max_iter'],  # 300 # default 300
        )

        if 'level' in self.settings.keys():
            level = self.settings['level']
        else:
            level = ['corpus', 'collection', 'language']

        if 'corpus' in level:
            create_plots_by_kmeans(
                dataset=dataset,
                level_to_plot='corpus',
                kmeans=kmeans,
                pd_x_scaled=pd_x_scaled,
                n_clusters=n_clusters,
                feat_name=feat_name,
                path_kmeans_feat=path_kmeans_feat,
                n_items = len(dataset.corpus.value_counts(dropna=False)),
                file_format_plots=file_format_plots
            )

        if sorted(dataset['collection'].unique()) != ['None'] and 'collection' in level:
            create_plots_by_kmeans(
                dataset=dataset,
                level_to_plot='collection',
                kmeans=kmeans,
                pd_x_scaled=pd_x_scaled,
                n_clusters=n_clusters,
                feat_name=feat_name,
                path_kmeans_feat=path_kmeans_feat,
                n_items=len(dataset.collection.value_counts(dropna=False)),
                file_format_plots=file_format_plots
            )

        if 'language' in level:
            create_plots_by_kmeans(
                dataset=dataset,
                level_to_plot='language',
                kmeans=kmeans,
                pd_x_scaled=pd_x_scaled,
                n_clusters=n_clusters,
                feat_name=feat_name,
                path_kmeans_feat=path_kmeans_feat,
                n_items=len(dataset.language.value_counts(dropna=False)),
                file_format_plots=file_format_plots
            )

        dataset['cluster'] = kmeans.labels_

        df_to_file(
            data=dataset,
            path_file=path_kmeans_feat + os.sep + feat_name + '_cluster_map',
            file_format_features='csv'
        )

        if 'full_cluster_map' in self.settings['k-means'].keys():
            if not self.settings['k-means']['full_cluster_map']:
                cluster_map_cut = dataset.drop(dataset.drop(['document', 'corpus', 'collection', 'language'], axis=1).columns, axis=1)
                cluster_map_cut['cluster'] = kmeans.labels_
                dataset = cluster_map_cut

                df_to_file(
                    data=cluster_map_cut,
                    path_file=path_kmeans_feat + os.sep + feat_name + '_cluster_map',
                    file_format_features='csv'
                )
        else:
            df_to_file(
                data=dataset,
                path_file=path_kmeans_feat + os.sep + feat_name + '_cluster_map',
                file_format_features='csv'
            )

        overview_cluster_per_dataset(
            data=dataset,
            feature=feat_name,
            path=path_kmeans_feat,
            file_format_plots=file_format_plots
        )

        logging.info('-----------------')
