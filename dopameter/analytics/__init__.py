import glob
import logging
import os
import pandas as pd

from dopameter.analytics.summarization import df_to_file
from dopameter.analytics.comparison import CompareAnalytics
from dopameter.analytics.vis_utils import matrix_to_boxplot


def detail_metrics(config):
    logging.info('===============================================================================================')
    logging.info('Get detailed features.')
    logging.info('===============================================================================================')

    feature_properties = {}
    feature_properties_parent = {}

    feature_properties_col = {}
    feature_properties_parent_col = {}

    corpora = set()
    collections = set()

    for feature in config['features']:

        for feature_file in glob.glob(config['output']['path_features'] + os.sep + '**/', recursive=True):

            for f in glob.glob(feature_file + os.sep + '*.csv'):

                feat = f.split(os.sep)[-2]

                if 'ngrams_tfidf' not in feat and feat == feature:
                    logging.info('Read feature [' + feat + '] file ' + f)

                    data = pd.read_csv(
                        filepath_or_buffer=f,
                        index_col=0,
                        keep_default_na=False,
                        low_memory=False
                    ).transpose().to_dict()
                    corp = os.path.basename(f).replace('.csv', '').replace('_' + f.split(os.sep)[-2], '')

                    if corp in config['corpora']:
                        corpora.add(corp)
                        for file in data:
                            for e in data[file]:
                                adapt_feat = feat + '__' + e
                                if adapt_feat not in feature_properties.keys():
                                    feature_properties[adapt_feat] = {}
                                    feature_properties_parent[adapt_feat] = feat

                                    if 'collection' in config['corpora'][corp]:
                                        feature_properties_col[adapt_feat] = {}
                                        feature_properties_parent_col[adapt_feat] = feat

                                if corp not in feature_properties[adapt_feat].keys():
                                    feature_properties[adapt_feat][corp] = []

                                    if 'collection' in config['corpora'][corp]:
                                        feature_properties_col[adapt_feat][config['corpora'][corp]['collection']] = []

                                if not feature_properties[adapt_feat][corp]:
                                    feature_properties[adapt_feat][corp] = [data[file][e]]

                                    if 'collection' in config['corpora'][corp]:
                                        feature_properties_col[adapt_feat][config['corpora'][corp]['collection']] = [data[file][e]]

                                else:
                                    temp = feature_properties[adapt_feat][corp]
                                    temp.append(data[file][e])
                                    feature_properties[adapt_feat][corp] = temp

                                    if 'collection' in config['corpora'][corp]:
                                        temp = feature_properties_col[adapt_feat][config['corpora'][corp]['collection']]
                                        temp.append(data[file][e])
                                        feature_properties_col[adapt_feat][config['corpora'][corp]['collection']] = temp

        path_features_detail = config['output']['path_features_detail']
        if not os.path.isdir(path_features_detail):
            os.mkdir(path_features_detail)

        path_features_detail_corpora = path_features_detail + os.sep + 'corpora'
        if not os.path.isdir(path_features_detail_corpora):
            os.mkdir(path_features_detail_corpora)

        path_features_detail_collections = path_features_detail + os.sep + 'collections'
        if not os.path.isdir(path_features_detail_collections):
            os.mkdir(path_features_detail_collections)

        logging.info('Detailed features in: ' + path_features_detail)

        for feat_prop in feature_properties:

            path_sub_feat_dir = path_features_detail_corpora + os.sep + feature_properties_parent[feat_prop]
            if not os.path.isdir(path_sub_feat_dir):
                os.mkdir(path_sub_feat_dir)

            path_sub_feat_file = path_sub_feat_dir + os.sep + feat_prop

            data = [feature_properties[feat_prop][d] for d in feature_properties[feat_prop]]
            index_labels = list(feature_properties[feat_prop].keys())

            for empty_corpus in [corpus for corpus in corpora if corpus not in feature_properties[feat_prop].keys()]:
                data.append([])
                index_labels.append(empty_corpus)

            df_to_file(
                data=pd.DataFrame(data, index=index_labels).sort_index(),
                path_file=path_sub_feat_file,
                file_format_features=config['settings']['file_format_features']
            )

        for feat_prop in feature_properties_col:

            path_sub_feat_dir = path_features_detail_collections + os.sep + feature_properties_parent_col[feat_prop]
            if not os.path.isdir(path_sub_feat_dir):
                os.mkdir(path_sub_feat_dir)

            path_sub_feat_file = path_sub_feat_dir + os.sep + feat_prop

            data = [feature_properties_col[feat_prop][d] for d in feature_properties_col[feat_prop]]
            index_labels = list(feature_properties_col[feat_prop].keys())

            for empty_corpus in [col for col in collections if col not in feature_properties_col[feat_prop].keys()]:
                data.append([])
                index_labels.append(empty_corpus)

            df_to_file(
                data=pd.DataFrame(data, index=index_labels).sort_index(),
                path_file=path_sub_feat_file,
                file_format_features=config['settings']['file_format_features']
            )



def visualization(config):
    logging.info('Tasks: plot')
    logging.info(config['output']['path_features_detail'])
    path_features_detail = config['output']['path_features_detail']

    for feature_file in glob.glob(path_features_detail + os.sep + '**/', recursive=True):

        for f in glob.glob(feature_file + os.sep + '*.csv'):

            if feature_file.split(os.sep)[len(feature_file.split(os.sep)) - 2] in config['features'].keys():
                data = pd.read_csv(
                    filepath_or_buffer=f,
                    index_col=0,
                    keep_default_na=False
                )
                data_bp = data.transpose().to_dict()

                matrix_to_boxplot(
                    title="Feature '" + os.path.basename(f).replace('.csv', '') + "'",
                    labels=data.index.values.tolist(),
                    file_plot=f.replace('.csv', ''),
                    data=pd.DataFrame([[float(data_bp[keys][vals]) for vals in data_bp[keys] if data_bp[keys][vals]] for keys in data_bp]).transpose(),
                    file_format_plots=config['settings']['file_format_plots'],
                    feature_cfg=os.path.basename(f).replace('.csv', ''),
                    height=int(config['settings']['boxplot_height'])
                )

def comparison(config):
    logging.info('===============================================================================================')
    logging.info('Comparison of corpora.')
    logging.info('===============================================================================================')

    analysis = CompareAnalytics(
        path_sources=config['output']['path_sources'],
        corpora=config['corpora'],
        path_compare=config['output']['path_compare'],
        compare_tasks=config['compare'],
        file_format_features=config['settings']['file_format_features'],
        most_frequent_words=config['settings']['most_frequent_words'],
        tasks=config['settings']['tasks']
    )

    analysis.source_analytics()