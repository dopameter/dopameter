import logging
import os

import pandas as pd
import seaborn as sns
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt


def get_colors(n):
    """get colors for scatter plots by given number colors

        Parameters
        ----------
         n: int

    """
    return [hsv_to_rgb(((1/n) * (i+0.5), 0.75, 0.75)) for i in range(n)]


def heatmap_cluster(data, title, path_fig):
    """ create heatmaps per dataset of aggregated datasets from t-sne or k-means clustering

        Parameters
        ----------
        data : dictionary or dataframe
        title : str
        path_fig: str
    """

    sns.set(
        rc={
            'figure.figsize': (len(data.columns), len(data.index)),
            'axes.grid': False,
            'xtick.bottom': False,
            'xtick.top': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.right': True,
            'axes.spines.top': True
        }
    )

    sns.heatmap(
        data,
        annot=True,
        cmap='YlOrBr',
        linewidth=1.5,
        fmt='',
        mask=data==0,
        square=True
    ).set(title=title, xlabel="Cluster", ylabel="Corpus")

    plt.title(label=title)
    plt.savefig(
        path_fig,
        bbox_inches='tight',
        format='png'
    )
    plt.close()


def heatmap_comparison(data, title, path_plot):
    """Draws a heatmap (using seaborn) of a given dictionary.

    Parameters
    ----------
    data : dataframe or dictionary
    title : str
    path_plot : str

    """
    sns.set(font_scale=2)
    sns.heatmap(
        pd.DataFrame(data).round(2),
        annot=True,
        cmap='crest',
        linewidth=.5,
        fmt=''
    ).set(title=title)

    size = len(data)*2 + 5
    sns.set(rc={'figure.figsize': (size, size)})
    plt.savefig(path_plot + '.png', bbox_inches='tight')
    plt.close()


def matrix_to_boxplot(title, labels, file_plot, data, file_format_plots, feature_cfg, height):
    """ dictionary with a 2-dimensional array into a boxplot

    Parameters
    ----------
    title : str
    labels : list of str
    file_plot : str
    data : dataframe
    file_format_plots : list of str
    feature_cfg : str
    height : int

    """

    feature = feature_cfg.split('__')[0]
    width = int(len(labels) / 2) + 6

    data.columns = labels
    ax = data.boxplot(column=labels, rot=30, fontsize=8, figsize=(width, height))
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_title(label=title)
    ax.set_xlabel(xlabel='corpus')

    if feature not in ['emotion', 'lexical_diversity', 'surface', 'syntax_dependency_metrics', 'syntax_constituency_metrics']:
        ax.set_ylabel('occurrences / document')
    else:
        ax.set_ylabel('score / document')

    fig_width, fig_height = plt.gcf().get_size_inches()
    if fig_width < 655 and fig_height < 655:
        for form in file_format_plots:
            plt.savefig(file_plot + '.' + form, bbox_inches='tight', format=form)
            logging.info('Boxplot: ' + file_plot + '.' + form)
    else:
        logging.warning(file_plot + '.png ' + ' too large.')
        logging.warning('No figure for feature ' + feature_cfg)
    plt.close()


def matrix_to_boxplot_alt(title, labels, file_plot, data, file_format_plots, feature_cfg, height):
    """ dictionary with a 2-dimensional array into a boxplot

    Parameters
    ----------
    title : str
    labels : list of str
    file_plot : str
    data : dictionary
    file_format_plots : list of str
    feature_cfg : str
    height : int

    """

    feature = feature_cfg.split('__')[0]
    width = int(len(labels) / 2) + 6
    fig, ax = plt.subplots(figsize=(width, height))

    ax.boxplot(data)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_title(label=title)
    ax.set_xlabel(xlabel='corpus')

    ax.set_xticklabels(labels=labels, rotation=30, fontsize=8)

    if feature not in ['emotion', 'lexical_diversity', 'surface', 'syntax_dependency_metrics', 'syntax_constituency_metrics']:
        ax.set_ylabel('occurrences / document')
    else:
        ax.set_ylabel('score / document')

    fig_width, fig_height = plt.gcf().get_size_inches()
    if fig_width < 655 and fig_height < 655:
        for form in file_format_plots:
            plt.savefig(file_plot + '.' + form, bbox_inches='tight', format=form)
            logging.info('Boxplot: ' + file_plot + '.' + form)
    else:
        logging.warning(file_plot + '.png ' + ' too large.')
        logging.warning('No figure for feature ' + feature_cfg)
    plt.close()


def overview_cluster_per_dataset(data, feature, path):
    """ create heatmaps per dataset of aggregated datasets from t-sne or k-means clustering

        Parameters
        ----------
        data : dataframe or dict
        feature: str
        path : str
    """

    corpora = set(data['corpus'].tolist())
    cluster = set(data['cluster'].tolist())
    dict_db = data[['cluster', 'corpus']].value_counts().to_dict()

    scores_tab = {}
    scores_tab_dev = {}
    for c in cluster:
        scores_tab[c] = {}
        scores_tab_dev[c] = {}
        for co in corpora:
            scores_tab[c][co] = 0
            scores_tab_dev[c][co] = 0

    freq = data['corpus'].value_counts().to_dict()
    for entry in dict_db:
        cluster, corpus = entry

        scores_tab[cluster][corpus] = int(dict_db[entry])
        scores_tab_dev[cluster][corpus] = int(dict_db[entry]) / freq[corpus]

    heatmap = pd.DataFrame.from_dict(scores_tab).sort_index().sort_index(axis=1).rename(columns={-1: 'Noise'})
    heatmap_freq = pd.DataFrame.from_dict(scores_tab_dev).sort_index().sort_index(axis=1).rename(columns={-1: 'Noise'}).round(3)

    heatmap_cluster(
        data=heatmap,
        title="occurrences per cluster ['" + feature + "']",
        path_fig= path + os.sep + 'occurrences_per_cluster.png'
    )
    heatmap_cluster(
        data=heatmap_freq,
        title="frequencies per cluster ['" + feature + "']",
        path_fig=path + os.sep + 'frequencies_per_cluster.png'
    )
