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


def heatmap_cluster(data, title, path_fig, file_format_plots):
    """ create heatmaps per dataset of aggregated datasets from t-sne or k-means clustering

        Parameters
        ----------
        data : dictionary or dataframe
        title : str
        path_fig: str
        file_format_plots : str
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
    for file_format in file_format_plots:
        plt.savefig(
            path_fig,
            bbox_inches='tight',
            format=file_format
        )
    plt.close()
    plt.clf()


def heatmap_comparison(data, title, path_plot, file_format_plots):
    """Draws a heatmap (using seaborn) of a given dictionary.

    Parameters
    ----------
    data : dataframe or dictionary
    title : str
    path_plot : str
    file_format_plots : [str]

    """
    sns.set(font_scale=3)
    sns.heatmap(
        data=pd.DataFrame(data).round(2),
        annot=True,
        cmap='YlOrBr',
        linewidth=.5,
        fmt='',
    ).set(
        title=title,
        xlabel='Source',
        ylabel='Target'
    )

    size = len(data)*2 + 5
    sns.set(rc={'figure.figsize': (size, size)})

    for file_format in file_format_plots:
        plt.savefig(path_plot + '.' + file_format, bbox_inches='tight')
    plt.close()
    plt.clf()


def matrix_to_boxplot(title, file_plot, data, file_format_plots, feature_cfg, height, xlabel, scale):
    """ dictionary with a 2-dimensional array into a boxplot

    Parameters
    ----------
    ----------
    title : str
    file_plot : str
    data : dataframe
    file_format_plots : list of str
    feature_cfg : str
    height : int
    xlabel : str
    scale : str

    """

    width = int(len(data.columns) / 2) + 6

    sns.set_theme(rc={
        'figure.figsize': (width, height)#,
    })

    sns.set(style="whitegrid")
    ax = sns.boxplot(data=data, palette="Spectral")

    if height > width:
        title = title.split(': ')[1]
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=(height*4/width)+10)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=(height*6/width)+10)
        ax.set_xlabel(xlabel=xlabel, fontsize=((height*4/width)+10)*0.75)

        ylabels = ax.yaxis.get_ticklabels()

        min = float(ylabels[1].get_text().replace("−", "-"))
        max = float(ylabels[len(ylabels)-2].get_text().replace("−", "-"))

        ax.text(x=0, y=min+((max-min)*0.01), s=title, rotation=90, fontsize=( (height*6/width)+10)*1.5, color='grey')

    else:
        fontsize = ((width*0.8/height)+10)
        ax.set_title(label=title, fontsize=fontsize*0.75)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fontsize*0.85)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize*0.85)
        ax.set_xlabel(xlabel=xlabel, fontsize=fontsize*0.75)

    ax.set_yscale(scale)

    fig_width, fig_height = plt.gcf().get_size_inches()
    if fig_width < 655 and fig_height < 655:
        for form in file_format_plots:
            plt.savefig(file_plot + '.' + form, bbox_inches='tight', format=form)
            logging.info('Boxplot: ' + file_plot + '.' + form)
    else:
        logging.warning(file_plot + ' too large.')
        logging.warning('No figure for feature ' + feature_cfg)
    plt.close()
    plt.clf()
    plt.cla()
    plt.close('all')


def overview_cluster_per_dataset(data, feature, path, file_format_plots):
    """ create heatmaps per dataset of aggregated datasets from t-sne or k-means clustering

        Parameters
        ----------
        data : dataframe or dict
        feature: str
        path : str
        file_format_plots : str
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

    for file_format in file_format_plots:
        heatmap_cluster(
            data=heatmap,
            title="occurrences per cluster ['" + feature + "']",
            path_fig=path + os.sep + 'occurrences_per_cluster.' + file_format,
            file_format_plots=file_format_plots
        )

    heatmap.to_excel(path + os.sep + 'occurrences_per_cluster.xlsx')
    heatmap.to_latex(path + os.sep + 'occurrences_per_cluster.tex')

    for file_format in file_format_plots:
        heatmap_cluster(
            data=heatmap_freq,
            title="frequencies per cluster ['" + feature + "']",
            path_fig=path + os.sep + 'frequencies_per_cluster.' + file_format,
            file_format_plots=file_format_plots
        )
    heatmap_freq.to_excel(path + os.sep + 'frequencies_per_cluster.xlsx')
    heatmap_freq.to_latex(path + os.sep + 'frequencies_per_cluster.tex')


def clean_file_name(file_name):

    """ remove characters from a file name

        Parameters
        ----------
        file_name : str
    """

    for c in ['<', '>', ':', '"', "\\", '|', '?', '*']:
        file_name = file_name.replace(c, '')
    return file_name


def create_corpus_analysis_plot(data, path_summary, file_format_plots, sum_type):

    path_summary = path_summary + os.sep + 'summary_plots'
    if not os.path.exists(path_summary):
        os.makedirs(path_summary)

    df_corpus_feats = data  ## ??

    df_corpus_feats.rename(columns={
        'characters': 'Zeichen',
        'types': 'Types / Vokabular',
        'tokens': 'Tokens',
        'sentences': 'Sätze',
        'documents': 'Dokumente',
        'lemmata': 'Lemmata',
        'different_sentences': 'Versch. Sätze'
    }, inplace=True)

    df_corpus_feats = df_corpus_feats.sort_index()

    fig_main, ax_main = plt.subplots()
    stacked_data = df_corpus_feats.transpose().apply(lambda x: x*100/sum(x), axis=1)
    stacked_data.plot(
        ax=ax_main,
        kind="barh",
        stacked=True,
        color=sns.color_palette("Spectral", len(df_corpus_feats)),
        edgecolor="dimgrey",
        figsize=(20, 6),
        xlabel="%"

    )

    handles, labels = ax_main.get_legend_handles_labels()

    if ax_main.get_legend() is not None:
        ax_main.get_legend().remove()

    figlegend = plt.figure(figsize=( (len( ''.join(df_corpus_feats.columns) )) , 2)) # Adjust size as needed
    figlegend.legend(handles, labels, loc='center', ncols=len(df_corpus_feats.index))

    for p_format in file_format_plots:
        fig_main.savefig(path_summary + os.sep + sum_type + '_characteristics_counts_plot.' + p_format, dpi=300)
    plt.close(fig_main)

    for p_format in file_format_plots:
        figlegend.savefig(path_summary + os.sep + sum_type + '_characteristics_counts_legend_across.' + p_format, dpi=300)

    figlegend = plt.figure(figsize=(4, len(df_corpus_feats) / 4)) # Adjust size as needed
    figlegend.legend(handles, labels, loc='center')
    for p_format in file_format_plots:
        figlegend.savefig(path_summary + os.sep + sum_type + '_characteristics_counts_legend_along.' + p_format, dpi=300)
    plt.close(figlegend)
    plt.clf()