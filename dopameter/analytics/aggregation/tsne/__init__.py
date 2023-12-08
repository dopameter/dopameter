import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

from dopameter.analytics.aggregation import ClusterCorpora
from dopameter.analytics.summarization import df_to_file
from dopameter.analytics.vis_utils import get_colors, overview_cluster_per_dataset


def create_plots(sns_rs, x_data, dataset, level_to_plot, labels, feat_name, path_tsne_feat, get_clusters):

    """ create plots for different level of composition of corpora, collection, languages - t-nse based plotting

        Parameters
        ----------
        sns_rs : dict
        x_data : numpy array
        dataset : dataframe
        level_to_plot : array of str
        labels : Any
        feat_name : str
        path_tsne_feat : str
        get_clusters : Any
    """

    path_tsne_feat = path_tsne_feat + os.sep + level_to_plot
    if not os.path.isdir(path_tsne_feat):
        os.mkdir(path_tsne_feat)

    colors = get_colors(n=len(set(labels)))

    for i, el in enumerate(set(labels)):

        sns.set(rc=sns_rs)
        sns.scatterplot(
            x=x_data[:, 0],
            y=x_data[:, 1],
            hue=dataset[level_to_plot],
            alpha=0.9,
            marker='o'
        ).set(xlabel='x_data', ylabel='y_data')

        plt.title(label="T-SNE - feature set '" + feat_name + "' [" + level_to_plot + ']')
        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            borderaxespad=0
        )
        plt.savefig(
            path_tsne_feat + os.sep + feat_name + '_' + level_to_plot + '_t-sne.png',
            bbox_inches='tight',
            format='png'
        )

        logging.info('Plot: ' + path_tsne_feat + os.sep + feat_name + '_' + level_to_plot + '_t-sne.png',)

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
            color='black',
            alpha=0.9,
            marker=marker
        )
        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            borderaxespad=0
        )

        plt.title(label="T-SNE - feature set '" + feat_name + "' [" + level_to_plot + '] ' + l)
        plt.savefig(
            path_tsne_feat + os.sep + feat_name + '_ ' + level_to_plot + ' _t-sne_' + l.replace(' ', '_') + '.png',
            bbox_inches='tight',
            format='png'
        )
        plt.close()
        plt.clf()

    if level_to_plot == 'corpus':
        iterate_over = sorted(dataset['corpus'].unique())
    elif level_to_plot == 'collection':
        iterate_over = sorted(dataset['collection'].unique())
    elif level_to_plot == 'language':
        iterate_over = sorted(dataset['language'].unique())

    for c in iterate_over:

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
        plt.title(label="t-SNE - feature set '" + feat_name + "' [" + level_to_plot + '] ')

        plt.savefig(
            path_tsne_feat + os.sep + feat_name + '_t-sne.png',
            bbox_inches='tight',
            format='png'
        )
        logging.info('Plot: ' + path_tsne_feat + os.sep + feat_name + '_t-sne.png')

        sns.set(rc=sns_rs)

        y = dataset[dataset[level_to_plot] == c][level_to_plot]

        sns.scatterplot(
            x=dataset[dataset[level_to_plot] == c]['x_data'],
            y=dataset[dataset[level_to_plot] == c]['y_data'],
            hue=y,
            palette=['black'],
            marker="2",
            s=25
        ).set(title="t-SNE - feature set '" + feat_name + "' [" + level_to_plot + '] ')

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


class ClusterTSNE(ClusterCorpora):

    """Get Aggregation and Clustering by T-SNE based plotting

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


    def cluster_tsne_by_feature(self, dataset, feat_name):
        """creates cluster of feature sets of t-sne and DBSCAN

        Parameters
        ----------

        dataset : dataframe
        feat_name : str

        """
        logging.info('================================================================================')
        logging.info('T-SNE Visualisation')
        logging.info("Feature '" + feat_name + "'")
        logging.info('================================================================================')

        dataset = dataset.fillna(0).rename(columns={'Unnamed: 0': 'document'})
        dataset = dataset.sort_values(by=['corpus'], ascending=True)

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
            self.settings['t-sne']['init'] = 'pca'
            self.settings['t-sne']['eps'] = 2  # maximum distance between two samples
            self.settings['t-sne']['min_samples'] = 10  # number of samples (or total weight) in a neighborhood

        if len(dataset) < int(self.settings['t-sne']['perplexity']):
            logging.warning('The length of your dataset is smaller than you configured perplexity: ' + str(len(dataset)) + ' < ' + self.settings['t-sne']['perplexity'])
            logging.warning('The configured perplexity is by length of your dataset!')
            self.settings['t-sne']['perplexity'] = float(len(dataset))

        logging.info("Compute T-SNE scores with input values " + str(self.settings['t-sne']))

        tsne = TSNE(
            n_components=int(self.settings['t-sne']['n_components']),
            random_state=int(self.settings['t-sne']['random_state']),
            perplexity=self.settings['t-sne']['perplexity'],
            learning_rate=self.settings['t-sne']['learning_rate'],
            init=self.settings['t-sne']['init']
        )

        x_data = tsne.fit_transform(
            X=dataset.drop(['document', 'corpus', 'collection', 'language'], axis=1)
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

            dataset['x_data'] = x_data[:, 0]
            dataset['y_data'] = x_data[:, 1]

            if 'level' in self.settings.keys():
                level = self.settings['level']
            else:
                level = ['corpus', 'collection', 'language']

            if 'corpus' in level:
                create_plots(
                    sns_rs=self.sns_rs,
                    x_data=x_data,
                    dataset=dataset,
                    level_to_plot='corpus',
                    labels=labels,
                    feat_name=feat_name,
                    path_tsne_feat=path_tsne_feat,
                    get_clusters=get_clusters
                )

            if sorted(dataset['collection'].unique()) != ['None'] and 'collection' in level:
                create_plots(
                    sns_rs=self.sns_rs,
                    x_data=x_data,
                    dataset=dataset,
                    level_to_plot='collection',
                    labels=labels,
                    feat_name=feat_name,
                    path_tsne_feat=path_tsne_feat,
                    get_clusters=get_clusters
                )

            if 'language' in level:
                create_plots(
                    sns_rs=self.sns_rs,
                    x_data=x_data,
                    dataset=dataset,
                    level_to_plot='language',
                    labels=labels,
                    feat_name=feat_name,
                    path_tsne_feat=path_tsne_feat,
                    get_clusters=get_clusters
                )

        tsne_frame = pd.DataFrame(x_data)
        tsne_frame['document'] = dataset['document'].values
        tsne_frame['corpus'] = dataset['corpus'].values
        tsne_frame['collection'] = dataset['collection'].values
        tsne_frame['language'] = dataset['language'].values
        tsne_frame['cluster'] = get_clusters

        df_to_file(
            data=tsne_frame,
            path_file=path_tsne_feat + os.sep + feat_name + '_tsne_dataset',
            file_format_features=self.file_format_features
        )

        overview_cluster_per_dataset(
            data=tsne_frame,
            feature=feat_name,
            path=path_tsne_feat
        )

        logging.info('-----------------')
