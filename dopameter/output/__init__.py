import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import logging
import seaborn as sns


def handle_features_output(path_features, feat, df_corpus_features, corpus, file_format_features, path_summary):
    """handle features output: write features of a set of metrics or counts in a table formatted file

    Parameters
    ----------
    path_features : str
    feat : str
    df_corpus_features : pandas dataframe
    corpus : corpus
    file_format_features : list of str
    path_summary : str

    """

    path_feature_bundle = path_features + os.sep + feat
    if not os.path.isdir(path_feature_bundle):
        os.mkdir(path_feature_bundle)

    write_df_to_file(
        data = df_corpus_features,
        path_file=path_feature_bundle + os.sep + corpus.name + '_' + feat,
        file_format_features=file_format_features
    )

    path_features_summary = path_summary + os.sep + feat
    if not os.path.isdir(path_features_summary):
        os.mkdir(path_features_summary)

    summary_df = pd.DataFrame()
    summary_df['max (document wise)'] = df_corpus_features.max()
    summary_df['mean (document wise)'] = df_corpus_features.mean()
    summary_df['min (document wise)'] = df_corpus_features.min()

    write_df_to_file(
        data=summary_df,
        path_file=path_features_summary + os.sep + corpus.name + '_' + feat + '_summary',
        file_format_features=file_format_features
    )

def draw_heatmap(data, title, path_plot):
    """Draws a heatmap (using seaborn) of a given dictionary.

    Parameters
    ----------
    data : dict
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

def write_df_to_file(data, path_file, file_format_features):
    """ write a pandas dataframe into a table formatted file

    Parameters
    ----------
    data : pandas dataframe
    path_file : str
    file_format_features : list of str

    """

    if 'csv' in file_format_features:
        data.to_csv(path_file + '.csv')
        logging.info('Table: ' + path_file + '.csv')
    if 'excel' in file_format_features or 'xlsx' in file_format_features:
        try:
            data.to_excel(path_file + '.xlsx', engine='openpyxl')
            logging.info('Table: ' + path_file + '.xlsx')
        except ValueError:
            logging.warning('Warning: ' + path_file + '.xlsx is not produced!')
            if 'csv' not in file_format_features:
                data.to_csv(path_file + '.csv')
                logging.info('Table: ' + path_file + '.csv')

        logging.info(path_file + '.xlsx')
    if 'latex' in file_format_features or 'tex' in file_format_features:
        f = open(path_file + '.tex', 'w')
        f.write(data.style.to_latex())
        f.close()
        logging.info('Table: '+ path_file + '.tex')
    if 'html' in file_format_features:
        data.to_html(path_file + '.html')
        logging.info('Table: ' + path_file + '.html')
    if set(['csv', 'excel', 'xlsx', 'tex', 'latex']).intersection(file_format_features) == "":
        logging.info("Wrong feature file format. Only 'csv', 'excel', 'xlsx', 'tex', 'latex' allowed.")


def write_macro_df_to_file(data, path_file, file_format_features):
    """ write a macro analysis into a file

    Parameters
    ----------
    data : dict
    path_file : str
    file_format_features : list of str

    """

    if 'csv' in file_format_features:
        if os.path.isfile(path_file + '.csv'):
            df = pd.read_csv(path_file + '.csv', index_col='corpus')
            data = pd.concat([pd.read_csv(io.StringIO(data.to_csv()), index_col='corpus'), df])
            data = (data.reset_index().drop_duplicates(keep='last').set_index('corpus').sort_index())
            data.to_csv(path_file + '.csv')
            logging.info('Table: ' + path_file + '.csv')
        else:
            write_df_to_file(data, path_file, ['csv'])

    if 'excel' in file_format_features or 'xlsx' in file_format_features:
        if os.path.isfile(path_file + '.xlsx'):
            df = pd.read_excel(path_file + '.xlsx', index_col='corpus', engine='openpyxl')
            data = pd.concat([data, df])
            data = (data.reset_index().drop_duplicates(keep='last').set_index('corpus').sort_index())
            data.to_excel(path_file + '.xlsx', engine='openpyxl')
            logging.info('Table: ' + path_file + '.xlsx')
            logging.warning('\tCheck the .xlsx output file - removing duplicates is not working well by .xlsx formats!',)
        else:
            write_df_to_file(data, path_file, ['xlsx'])

    if 'latex' in file_format_features or 'tex' in file_format_features:
        write_df_to_file(data, path_file, ['tex'])
        logging.warning('\tIf there has been an old latex file, it is overwritten.')
    if 'html' in file_format_features:
        write_df_to_file(data, path_file, ['html'])
        logging.warning('\tIf there has been an old html file, it is overwritten.')


def boxplot_2dim_array(title, labels, file_plot, data, file_format_plots, feature_cfg, height):
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

    if feature not in ['emotion', 'lexical_richness', 'surface', 'syntax_dependency_metrics', 'syntax_constituency_metrics']:
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

def sum_all_corpora(path_counts, sum_all_corpora_counts, feature_name, file_format_features):
    """write scores of summation of all corpora
    Parameters
    ----------
    path_counts : str
    sum_all_corpora_counts : str
    feature_name : str
    file_format_features : str
    """

    data = pd.DataFrame(sum_all_corpora_counts).transpose()
    data.index.name = 'corpus'

    write_macro_df_to_file(
        data=data,
        path_file=path_counts + os.sep + 'corpora_counts_' + feature_name,
        file_format_features=file_format_features
    )


def write_macro_features(path_summary, corpus, feature_name, file_format_features):
    """write macro feature tables

    Parameters
    ----------
    path_summary : str
    corpus : corpus
    feature_name : str
    file_format_features : str
    """

    macro_corpus_feats = pd.DataFrame({corpus.name: corpus.macro_features}).transpose().to_dict()

    if feature_name == 'corpus_characteristics':
        macro_corpus_feats_file = path_summary + os.sep + 'corpora_characteristics_counts'
        df_corpus_feats = pd.DataFrame(
            macro_corpus_feats[feature_name],
            index=macro_corpus_feats[feature_name][list(macro_corpus_feats[feature_name].keys())[0]].keys()
        ).transpose()
    else:
        macro_corpus_feats_file = path_summary + os.sep + 'corpora_characteristics_' + feature_name
        filtered_dict = {c: macro_corpus_feats[feature_name][c]['features'] for c in macro_corpus_feats[feature_name]}
        df_corpus_feats = pd.DataFrame(filtered_dict).transpose()

    df_corpus_feats.index.name = 'corpus'
    write_macro_df_to_file(
        data=df_corpus_feats,
        path_file=macro_corpus_feats_file,
        file_format_features=file_format_features
    )
