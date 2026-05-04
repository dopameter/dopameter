import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

from dopameter.analytics.aggregation import ClusterCorpora
from dopameter.analytics.summarization import df_to_file
from dopameter.analytics.vis_utils import get_colors, overview_cluster_per_dataset, clean_file_name
#from dopameter.analytics.vis_utils import overview_cluster_per_dataset, clean_file_name


def create_plots_tsne(
        sns_rs,
        x_data,
        dataset,
        level_to_plot,
        labels,
        feat_name,
        path_tsne_feat,
        #db_scan_clusters,
        plot_config,
        n_items,
        file_format_plots
    ):

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
        plot_config : dict
        n_items : int

        Note
        ----
        Details of plot_config, see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

    """

    path_tsne_feat = path_tsne_feat + os.sep + level_to_plot
    if not os.path.isdir(path_tsne_feat):
        os.mkdir(path_tsne_feat)

    n = len(set(labels))
    colors = (sns.color_palette("flare", int(n / 2)) +
              sns.color_palette("viridis", int(n / 2) + int(n % 2)))

    if level_to_plot == 'corpus':
        iterate_over = sorted(dataset['corpus'].unique())
    elif level_to_plot == 'collection':
        iterate_over = sorted(dataset['collection'].unique())
    elif level_to_plot == 'language':
        iterate_over = sorted(dataset['language'].unique())
    else:
        exit(0)

    n = len(set(iterate_over))
    colors2 = (sns.color_palette("flare", int(n / 2)) + sns.color_palette("viridis", int(n / 2) + int(n % 2)))

    sns.scatterplot(
        x=x_data[:, 0],
        y=x_data[:, 1],
        hue=dataset[level_to_plot],
        alpha=float(plot_config['alpha']),
        marker=plot_config['marker'],
        s=float(plot_config['s']),
        palette=colors2,
    ).set(xlabel='t-SNE 1', ylabel='t-SNE 2')


    plt.title(label="T-SNE - feature set '" + feat_name + "' [" + level_to_plot + ']')
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0,
        fontsize=plot_config['font_size'],
        markerscale=plot_config['markerscale']
    )

    for file_format in file_format_plots:
        plt.savefig(
            path_tsne_feat + os.sep + clean_file_name(feat_name + '_' + level_to_plot + '_t-sne.' + file_format),
            bbox_inches='tight',
            format=file_format
        )
        logging.info('(0) Plot: ' + path_tsne_feat + os.sep + clean_file_name(feat_name + '_' + level_to_plot + '_t-sne.' + file_format))
    plt.close()
    plt.clf()
    plt.cla()
    plt.close('all')

    for i, el in enumerate(set(labels)):

        if 'detail_plots' in plot_config.keys():
            if plot_config['detail_plots'] == 'true':

                if el != -1:
                    l = 'Cluster ' + str( (el+1) )
                    marker = '2'
                else:
                    l = 'Noise'
                    marker = '^'

                plt.scatter(
                    x_data[labels == el, 0],
                    x_data[labels == el, 1],
                    label=l,
                    color='black',
                    alpha=float(plot_config['alpha']),
                    s=float(plot_config['s']),
                    marker=marker
                )
                plt.legend(
                    bbox_to_anchor=(1.02, 1),
                    loc='upper left',
                    borderaxespad=0,
                    fontsize=plot_config['font_size'],
                    markerscale=plot_config['markerscale']
                )

                plt.title(label="T-SNE - feature set '" + feat_name + "' [" + level_to_plot + '] ' + l)

                for file_format in file_format_plots:
                    plt.savefig(
                        path_tsne_feat + os.sep + clean_file_name(feat_name + '_ ' + level_to_plot + ' _t-sne_' + l.replace(' ', '_') + '.' + file_format),
                        bbox_inches='tight',
                        format=file_format
                    )
                plt.close()
                plt.clf()
                plt.cla()
                plt.close('all')

    for c in iterate_over:

        for i, el in enumerate(set(labels)):

            if el != -1:
                l = 'Cluster ' + str( (el+1) )
                marker = plot_config['marker']
            else:
                l = 'Noise'
                marker = '<'

            plt.scatter(
                x_data[labels == el, 0],
                x_data[labels == el, 1],
                label=l,
                color=colors[i],
                alpha=float(plot_config['alpha']),
                s=float(plot_config['s']),
                marker=marker
            )
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')

        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            borderaxespad=0,
            fontsize=plot_config['font_size'] * 2,
            markerscale=plot_config['markerscale']
        )
        plt.title(label="t-SNE - feature set '" + feat_name + "' [" + level_to_plot + '] ')

        for file_format in file_format_plots:
            plt.savefig(
                path_tsne_feat + os.sep + feat_name + '_t-sne.' + file_format,
                bbox_inches='tight',
                format=file_format
            )
            logging.info('(1) Plot: ' + path_tsne_feat + os.sep + feat_name + '_t-sne.' + file_format)

        y = dataset[dataset[level_to_plot] == c][level_to_plot]

        sns.scatterplot(
            x=dataset[dataset[level_to_plot] == c]['x_data'],
            y=dataset[dataset[level_to_plot] == c]['y_data'],
            hue=y,
            palette=['black'],
            marker="2"
        ).set(title="t-SNE - feature set '" + feat_name + "' [" + level_to_plot + '] ')

        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            borderaxespad=0,
            fontsize=plot_config['font_size']
        )

        for file_format in file_format_plots:
            plt.savefig(
                path_tsne_feat + os.sep + clean_file_name(feat_name + '_tsne_' + c + '.' + file_format),
                bbox_inches='tight',
                format=file_format
            )
            logging.info('(2) Plot: ' + path_tsne_feat + os.sep + clean_file_name(feat_name + '_tsne_' + c + '.' + file_format))

        plt.close()
        plt.clf()
        plt.cla()
        plt.close('all')

    plt.close()
    plt.clf()
    plt.cla()
    plt.close('all')


# TODO umbenennen, das Cluster ist DBSCAN
class ClusterDBSCAN(ClusterCorpora):

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
        self.algorithm = None

    def cluster_tsne_by_feature(self, dataset, feat_name, file_format_plots):
        """creates cluster of feature sets of t-sne and DBSCAN

        Parameters
        ----------

        dataset : dataframe
        feat_name : str
        file_format_plots : [str]

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

        if "k-means" in self.settings.keys():
            self.algorithm = "k-means"
        else:
            self.algorithm = "DBSCAN"

        if len(dataset) < int(self.settings['t-sne']['perplexity']):
            logging.warning('The length of your dataset is smaller than you configured perplexity: ' + str(len(dataset)) + ' < ' + self.settings['t-sne']['perplexity'])
            logging.warning('The configured perplexity is by length of your dataset!')
            self.settings['t-sne']['perplexity'] = float(len(dataset))

        logging.info("Compute T-SNE plot with input configuration " + str(self.settings['t-sne']))

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

        y = dataset['corpus']
        if self.algorithm == 'k-means':
            kmeans = KMeans(
                #n_clusters=2,  # self.settings['k-means']['n_clusters'],  # n_clusters,
                n_clusters=self.settings['k-means']['n_clusters'],  # self.settings['k-means']['n_clusters'],  # n_clusters,
                random_state=None,  # self.settings['k-means']['random_state'],  # None,#1,
                n_init="auto",  # self.settings['k-means']['n_init'],  # 'auto',# default 10  #20,
                max_iter=300,  # self.settings['k-means']['max_iter'],  # 300 # default 300
            )

            pd_x_scaled = pd.DataFrame(
                preprocessing.MinMaxScaler().fit_transform(
                    dataset.drop(['document', 'corpus', 'collection', 'language'], axis=1)),
                columns=dataset.drop(['document', 'corpus', 'collection', 'language'], axis=1).columns
            )

            y = dataset['corpus']
            y_data = kmeans.fit_transform(X=pd_x_scaled, y=LabelEncoder().fit_transform(y))
            labels = kmeans.labels_

        else:
            db = DBSCAN(
                eps=self.settings['t-sne']['eps'],
                min_samples=self.settings['t-sne']['min_samples']
            )

            db_scan_clusters = db.fit_predict(x_data)
            db_scan_clusters = db.fit_predict(x_data)

            labels = db.fit(x_data).labels_

            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            logging.info("Estimated number of clusters: %d" % n_clusters_)
            logging.info("Estimated number of noise points: %d" % n_noise_)

        if 'plot' in self.tasks:

            dataset['x_data'] = x_data[:, 0]
            dataset['y_data'] = x_data[:, 1]

            if 'level' in self.settings.keys():
                level = self.settings['level']
            else:
                level = ['corpus', 'collection', 'language']

            if 'plot' in self.settings['t-sne'].keys():
                plot_config = self.settings['t-sne']['plot']
            else:
                plot_config = {
                    "alpha": "0.75", # blending value, between 0 (transparent) and 1 (opaque)
                    "marker": "o",  # marker
                    "s": "20",        # marker size
                    "font_size": 10,
                    "markerscale": 2,
                }

            if 'corpus' in level:
                create_plots_tsne(
                    sns_rs=self.sns_rs,
                    x_data=x_data,
                    dataset=dataset,
                    level_to_plot='corpus',
                    labels=labels,
                    feat_name=feat_name,
                    path_tsne_feat=path_tsne_feat,
                    #db_scan_clusters=db_scan_clusters,
                    plot_config=plot_config,
                    n_items=len(dataset.corpus.value_counts(dropna=False)),
                    file_format_plots=file_format_plots
                )

            if sorted(dataset['collection'].unique()) != ['None'] and 'collection' in level:
                create_plots_tsne(
                    sns_rs=self.sns_rs,
                    x_data=x_data,
                    dataset=dataset,
                    level_to_plot='collection',
                    labels=labels,
                    feat_name=feat_name,
                    path_tsne_feat=path_tsne_feat,
                    #db_scan_clusters=db_scan_clusters,
                    plot_config=plot_config,
                    n_items=len(dataset.collection.value_counts(dropna=False)),
                    file_format_plots=file_format_plots
                )

            if 'language' in level:
                create_plots_tsne(
                    sns_rs=self.sns_rs,
                    x_data=x_data,
                    dataset=dataset,
                    level_to_plot='language',
                    labels=labels,
                    feat_name=feat_name,
                    path_tsne_feat=path_tsne_feat,
                    #db_scan_clusters=db_scan_clusters,
                    plot_config=plot_config,
                    n_items=len(dataset.language.value_counts(dropna=False)),
                    file_format_plots=file_format_plots
                )

        tsne_frame = pd.DataFrame(x_data)

        tsne_frame['document']   = dataset['document'].values
        tsne_frame['corpus']     = dataset['corpus'].values
        tsne_frame['collection'] = dataset['collection'].values
        tsne_frame['language']   = dataset['language'].values
        tsne_frame['cluster']    = labels + 1

        df_to_file(
            data=tsne_frame,
            path_file=path_tsne_feat + os.sep + feat_name + '_cluster_map',
            file_format_features=self.file_format_features
        )

        overview_cluster_per_dataset(
            data=tsne_frame,
            feature=feat_name,
            path=path_tsne_feat,
            file_format_plots=file_format_plots
        )

        logging.info('-----------------')
