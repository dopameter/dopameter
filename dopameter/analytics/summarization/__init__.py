import pandas as pd
import os
import io
import logging

from dopameter.configuration.installation import ConfLanguages


def features_to_file(path_features, feat, df_corpus_features, corpus, file_format_features):
    """features output: write features of a set of metrics or counts in a table formatted file

    Parameters
    ----------
    path_features : str
    feat : str
    df_corpus_features : pandas dataframe
    corpus : corpus
    file_format_features : list of str

    """

    path_feature_bundle = path_features + os.sep + feat
    if not os.path.isdir(path_feature_bundle):
        os.mkdir(path_feature_bundle)

    df_to_file(
        data = df_corpus_features,
        path_file=path_feature_bundle + os.sep + corpus.name + '_' + feat,
        file_format_features=file_format_features
    )


def df_to_file(data, path_file, file_format_features):
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
    if {'csv', 'excel', 'xlsx', 'tex', 'latex'}.intersection(file_format_features) == "":
        logging.info("Wrong feature file format. Only 'csv', 'excel', 'xlsx', 'tex', 'latex' allowed.")


def macro_df_to_file(data, path_file, file_format_features):
    """ write a macro analysis into a file

    Parameters
    ----------
    data : pandas dataframe
    path_file : str
    file_format_features : str

    """

    if 'csv' in file_format_features:
        if os.path.isfile(path_file + '.csv'):
            df = pd.read_csv(path_file + '.csv', index_col='corpus')
            data = pd.concat([pd.read_csv(io.StringIO(data.to_csv()), index_col='corpus'), df])
            data = (data.reset_index().drop_duplicates(keep='last').set_index('corpus').sort_index())
            data.to_csv(path_file + '.csv')
            logging.info('Table: ' + path_file + '.csv')
        else:
            df_to_file(
                data=data,
                path_file=path_file,
                file_format_features=['csv']
            )

    if 'excel' in file_format_features or 'xlsx' in file_format_features:
        if os.path.isfile(path_file + '.xlsx'):
            df = pd.read_excel(io=path_file + '.xlsx', index_col='corpus', engine='openpyxl')
            data = pd.concat([data, df])
            data = (data.reset_index().drop_duplicates(keep='last').set_index('corpus').sort_index())
            data.to_excel(io=path_file + '.xlsx', engine='openpyxl')
            logging.info('Table: ' + path_file + '.xlsx')
            logging.warning('\tCheck the .xlsx output file - removing duplicates is not working well by .xlsx formats!',)
        else:
            df_to_file(
                data=data,
                path_file=path_file,
                file_format_features=['xlsx']
            )

    if 'latex' in file_format_features or 'tex' in file_format_features:
        df_to_file(
            data=data,
            path_file=path_file,
            file_format_features=['tex']
        )
        logging.warning('\tIf there has been an old latex file, it is overwritten.')
    if 'html' in file_format_features:
        df_to_file(
            data=data,
            path_file=path_file,
            file_format_features=['html']
        )
        logging.warning('\tIf there has been an old html file, it is overwritten.')


def sum_all_corpora_to_file(path_counts, sum_all_corpora_counts, feature_name, file_format_features, corpus_name, collection, language):
    """write scores of summation of all corpora
    Parameters
    ----------
    path_counts : str
    sum_all_corpora_counts : dict
    feature_name : str
    file_format_features : str
    corpus_name : str or list
    collection : str or list
    language : str
    """

    data = pd.DataFrame(sum_all_corpora_counts).transpose()

    if type(collection) == list: # language
        data.insert(loc=0, column='bundle of collections', value=str(collection))
        data.insert(loc=0, column='bundle corpora', value=str(corpus_name))
        path_counts = path_counts + os.sep + 'counts_language'
        if not os.path.isdir(path_counts):
            os.mkdir(path_counts)
        path_file = path_counts + os.sep + 'language_counts_' + feature_name
    elif type(corpus_name) == list: # collection
        data.insert(loc=0, column='language', value=language)
        data.insert(loc=0, column='bundle of corpora', value=corpus_name)
        path_counts = path_counts + os.sep + 'counts_collection'
        if not os.path.isdir(path_counts):
            os.mkdir(path_counts)
        path_file = path_counts + os.sep + 'collection_counts_' + feature_name
    else: # corpus
        data.insert(loc=0, column='language', value=language)
        data.insert(loc=0, column='collection', value=collection)
        path_counts = path_counts + os.sep + 'counts_corpora'
        if not os.path.isdir(path_counts):
            os.mkdir(path_counts)
        path_file = path_counts + os.sep + 'corpora_counts_' + feature_name

    data.index.name = 'corpus'

    macro_df_to_file(
        data=data,
        path_file=path_file,
        file_format_features=file_format_features
    )


def macro_features_to_file(path_summary, corpus, feature_name, file_format_features):
    """write macro feature tables

    Parameters
    ----------
    path_summary : str
    corpus : corpus
    feature_name : str
    file_format_features : str
    """

    macro_corpus_feats = pd.DataFrame({corpus.name: corpus.macro_features}).transpose().to_dict()

    path_sum_lang = path_summary + os.sep + 'macro_scores_language'
    if not os.path.isdir(path_sum_lang):
        os.mkdir(path_sum_lang)
    path_sum_coll = path_summary + os.sep + 'macro_scores_collections'
    if not os.path.isdir(path_sum_coll):
        os.mkdir(path_sum_coll)
    path_sum_corp = path_summary + os.sep + 'macro_scores_corpora'
    if not os.path.isdir(path_sum_corp):
        os.mkdir(path_sum_corp)

    if feature_name == 'corpus_characteristics':
        macro_corpus_feats_file = path_summary + os.sep + 'corpora_characteristics_counts'
        df_corpus_feats = pd.DataFrame(
            macro_corpus_feats[feature_name],
            index=macro_corpus_feats[feature_name][list(macro_corpus_feats[feature_name].keys())[0]].keys()
        ).transpose()
    else:

        filtered_dict = {c: macro_corpus_feats[feature_name][c]['features'] for c in macro_corpus_feats[feature_name]}

        if not hasattr(corpus, 'collection_name') and not hasattr(corpus, 'collections'):
            filtered_dict = {k : filtered_dict[k] for k in filtered_dict}
            macro_corpus_feats_file = path_sum_coll + os.sep + 'collection_characteristics_' + feature_name
        elif hasattr(corpus, 'collections'):
            filtered_dict = {k:filtered_dict[k] for k in filtered_dict }
            macro_corpus_feats_file = path_sum_lang + os.sep + 'language_characteristics_' + feature_name
        else:
            macro_corpus_feats_file = path_sum_corp + os.sep + 'corpora_characteristics_' + feature_name

        df_corpus_feats = pd.DataFrame(filtered_dict).transpose()

    if hasattr(corpus, 'collection_name'): # corpus
        df_corpus_feats.insert(loc=0, column='collection', value=corpus.collection_name)
    else:
        if hasattr(corpus, 'collections') and corpus.corpora:
            df_corpus_feats.insert(loc=0, column='collection of corpora', value=str([cor.name for cor in corpus.corpora]))


    if hasattr(corpus, 'collections'):
        df_corpus_feats.insert(loc=0, column='bundle of collections', value=str([cor for cor in corpus.collections]))
        df_corpus_feats.insert(loc=0, column='bundle of corpora', value=str([corp.name for col in corpus.collections for corp in corpus.collections[col].corpora]))

    else:
        # collection
        if not hasattr(corpus, 'collection_name') and not hasattr(corpus, 'collections'):
            df_corpus_feats.insert(loc=0, column='language', value=ConfLanguages().lang_def[corpus.lang])
            df_corpus_feats.insert(loc=0, column='bundle of corpora ', value=str([col.name for col in corpus.corpora]))
        else:
            # corpus
            df_corpus_feats.insert(loc=0, column='language', value=ConfLanguages().lang_def[corpus.lang])

    df_corpus_feats.index.name = 'corpus'

    if feature_name in ['emotion', 'lexical_diversity', 'surface', 'ner', 'pos', 'token_characteristics']:
        df_corpus_feats = df_corpus_feats.round(4)

    macro_df_to_file(
        data=df_corpus_feats,
        path_file=macro_corpus_feats_file,
        file_format_features=file_format_features
    )
