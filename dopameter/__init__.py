import glob
import json
import os
import numpy as np
import pandas as pd
import collections
import logging

from sklearn.feature_extraction.text import TfidfVectorizer

from dopameter.configuration.installation import ConfLanguages
from dopameter.output import write_df_to_file, write_macro_df_to_file, boxplot_2dim_array, sum_all_corpora, \
    write_macro_features, handle_features_output
from dopameter.compare import CompareAnalytics


class DoPaMeter:
    """DoPaMeter

    Parameters
    ----------

    config : dict


    Notes
    -----

    Here, all fucntionalities of DoPa Meter are convergenced.
    For a deeper knowledge, read the code documentation under the path 'doc'.
    This class represents an automat, that combines all functionalities of DoPa Meter.
    It is derived into sections of 'feautres', 'counts' and 'corpus_characteristics',
    that is computed by a run trough a nlp pipeline and a lot of modules of a hub o features.
    followed by 'plot', 'features_detail' and 'clusters'.
    """

    def __init__(
            self,
            config
    ):
        self.config = config

    def run_dopameter(self):

        tasks = self.config['settings']['tasks']
        logging.info('tasks: ' + str(tasks))

        valid_tasks = {
            'features',
            'features_detail',
            'plot',
            'counts',
            'corpus_characteristics',
            'compare',
            'cluster'
        }

        if not set(tasks).intersection(valid_tasks):
            exit("Your given set of tasks is wrong. Allowed task definitions: " + ' '.join(valid_tasks) + ".")

        file_format_features = self.config['settings']['file_format_features']
        file_format_plots = self.config['settings']['file_format_plots']

        if 'compare' in tasks and 'features' not in tasks and 'counts' not in tasks and 'corpus_characteristics' not in tasks:
            if set(self.config['compare']).intersection({'bleu', 'nist', 'meteor'}):
                self.config['features'] = 'bleu'
                tasks += 'features'

        if set(tasks).intersection({'features', 'counts', 'corpus_characteristics'}):

            path_summary = self.config['output']['path_summary']
            if not os.path.isdir(path_summary):
                os.mkdir(path_summary)

            if 'counts' in tasks or 'corpus_characteristics' in tasks:
                path_counts = self.config['output']['path_counts']
                if not os.path.isdir(path_counts):
                    os.mkdir(path_counts)

            if 'features' in tasks or 'counts' in tasks:
                features = self.config['features']
                if not features:
                    raise ValueError(
                        "Stop running task 'features'! - No features defined! Define 'features' in your config file!")
                else:
                    logging.info('features: ' + str(features))

                if 'external_resources' in self.config.keys():
                    external_resources = self.config['external_resources']
                    logging.info('external resources: ')

                    for ext in external_resources:
                        if type(external_resources[ext]) == dict:
                            for dictionary in external_resources[ext]:
                                logging.info('\t - ' + dictionary + ': ' + external_resources[ext][dictionary])
                        else:
                            logging.info('\t - ' + ext + ': ' + external_resources[ext])
                else:
                    logging.info('no external resources')
                    external_resources = {}

                if bool(self.config['settings']['store_sources']):
                    path_storage_sources = self.config['output']['path_sources']
                    if not os.path.isdir(path_storage_sources):
                        os.mkdir(path_storage_sources)

                if 'features' in tasks:
                    path_features = self.config['output']['path_features']
                    if not os.path.isdir(path_features):
                        os.mkdir(path_features)

                    if 'ngrams' in features.keys():
                        features['ngrams'] = [int(i) for i in features['ngrams']]
            else:
                features = {}
                external_resources = {}

            from dopameter.corpora import SetUpCorpora
            corpora = SetUpCorpora(
                corpora=self.config['corpora'],
                features=features,
                external_resources=external_resources
            ).create_corpora()

            corpora_names = [corpus.name for corpus in corpora]

            if len(corpora_names) > 1:
                logging.info(str(len(corpora_names)) + ' corpora to process: ' + str(corpora_names))
            elif len(corpora_names) == 1:
                logging.info(str(len(corpora_names)) + ' corpus to process: ' + str(corpora_names))
            else:
                exit('Routine aborted! There are no corpora selected. Configure minimal 1 text corpus with files!')

            langs = set([self.config['corpora'][c]['language'] for c in self.config['corpora']])

            from dopameter.configuration.pipeline import PreProcessingPipline
            pipeline = PreProcessingPipline()

            corp_i = 0
            for lang in langs:
                logging.info("Start to process language '" + ConfLanguages().lang_def[lang] + "'")
                nlp = pipeline.create_nlp(lang)

                from dopameter.corpora.corpus.document import BasicCharacteristics
                basic_chars = BasicCharacteristics()

                if 'token_characteristics' in features.keys():
                    from dopameter.features.token_characteristics import TokenCharacteristics
                    tok_chars = TokenCharacteristics(
                        features=features['token_characteristics'],
                    )

                if 'ner' in features.keys():
                    from dopameter.features.ner import NERFeatures
                    ner = NERFeatures(
                        nlp=nlp,
                        features=features['ner']
                    )

                if 'pos' in features.keys():
                    from dopameter.features.pos import POSFeatures
                    pos = POSFeatures(
                        nlp=nlp,
                        features=features['pos']
                    )

                if 'ngrams' in features.keys():
                    from dopameter.features.ngrams import NGramFeatures
                    ngrams = NGramFeatures(features=features['ngrams'])
                    features['ngrams'] = [int(i) for i in features['ngrams']]

                if 'dictionary_lookup' in features.keys():
                    from dopameter.features.semantics.dictionary_lookup import DictionaryLookUp
                    if 'dictionary_lookup' in features.keys():
                        dictionary_lookup = DictionaryLookUp(
                            nlp=nlp,
                            path_dictionaries=self.config['external_resources']['dictionaries'],
                            file_format_dicts=self.config['settings']['file_format_dicts'],
                            features=features['dictionary_lookup']
                        )

                if 'lexical_richness' in features.keys():
                    from dopameter.features.lexical_richness import LexicalRichnessFeatures
                    lexical_richness = LexicalRichnessFeatures(
                        features=features['lexical_richness']
                    )

                if 'surface' in features.keys():
                    import spacy_syllables  # Do not delete this line!
                    nlp.add_pipe("syllables", after="tagger")
                    if lang == 'de':
                        from dopameter.features.surface_readability.de import SurfaceFeaturizesDE
                        surface = SurfaceFeaturizesDE(
                            features=features['surface']
                        )
                    elif lang == 'en':
                        from dopameter.features.surface_readability.en import SurfaceFeaturizesEN
                        surface = SurfaceFeaturizesEN(
                            features=features['surface']
                        )
                    else:
                        from dopameter.features.surface_readability import SurfaceFeaturizes
                        surface = SurfaceFeaturizes(
                            features=features['surface']
                        )

                if 'syntax_dependency_metrics' in features.keys() or 'syntax_dependency_tree' in features.keys():
                    from dopameter.features.syntax.dependency import DependencyFeatures
                    syntax_dependency = DependencyFeatures(
                        features=features
                    )

                if 'syntax_constituency_metrics' in features.keys() or 'syntax_constituency_tree' in features.keys():
                    from dopameter.features.syntax.constituency import ConstituencyFeatures
                    syntax_constituency = ConstituencyFeatures(
                        lang=lang,
                        features=features
                    )

                if ('wordnet_synsets' in features.keys() or
                        'wordnet_senses' in features.keys() or
                        'wordnet_semantic_relations' in features.keys()
                ):
                    from dopameter.features.semantics.wordnet_relations import WordNetFeatures
                    semantics_wordnet = WordNetFeatures(
                        lang=lang,
                        features=features
                    )

                if 'emotion' in features.keys():
                    from dopameter.features.emotion import EmotionFeatures
                    emotion = EmotionFeatures(
                        lang=lang,
                        features=features['emotion']
                    )

                logging.info(
                    '===============================================================================================')
                logging.info('Initialization of languages and feature modules done')
                logging.info(
                    '===============================================================================================')
                logging.info('Start to process corpora - language: ' + ConfLanguages().lang_def[lang])
                logging.info(
                    '===============================================================================================')

                for corpus in [corpus for corpus in corpora if corpus.lang == lang]:
                    corp_i += 1
                    logging.info(
                        '-----------------------------------------------------------------------------------------------')
                    logging.info(
                        '\tStart to process corpus (' + str(corp_i) + '/' + str(len(corpora)) + '): ' + corpus.name)
                    logging.info('\t# ' + str(len(corpus.files)) + ' files from ' + corpus.path)
                    logging.info(
                        '-----------------------------------------------------------------------------------------------')

                    for i, f in enumerate(corpus.files):
                        logging.info("process corpus '" + corpus.name + "' (" + str(corp_i) + '/' + str(
                            len(corpora)) + ') - file (' + str(i + 1) + '/' + str(len(corpus.files)) + ') ' + str(f))

                        doc_name = os.path.basename(f)
                        plain_text = open(file=f, encoding=corpus.encoding).read()
                        doc = nlp(plain_text)

                        if doc._.n_tokens == 0:
                            logging.warning('File ' + f + ' has 0 tokens. The document is not processed!')

                        elif doc._.n_tokens > 0:

                            corpus.update_count_characteristics(
                                cnt=basic_chars.count_doc(doc=doc),
                                doc_name=doc_name
                            )

                            if 'ner' in features.keys():
                                corpus.update_properties(
                                    feature='ner',
                                    data=ner.feat_doc(doc=doc),
                                    doc_name=doc_name
                                )

                            if 'pos' in features.keys():
                                corpus.update_properties(
                                    feature='pos',
                                    data=pos.feat_doc(doc=doc),
                                    doc_name=doc_name
                                )

                            if 'token_characteristics' in features.keys():
                                corpus.update_properties(
                                    feature='token_characteristics',
                                    data=tok_chars.feat_doc(doc=doc),
                                    doc_name=doc_name
                                )

                            if 'ngrams' in features.keys():
                                corpus.update_properties(
                                    feature='ngrams',
                                    data=ngrams.feat_doc(doc=doc),
                                    doc_name=doc_name
                                )

                            if 'dictionary_lookup' in features.keys():
                                dictionary_lookup_feats = dictionary_lookup.feat_doc(doc=doc)
                                for feature_file in dictionary_lookup_feats.keys():
                                    corpus.update_properties(
                                        feature=feature_file,
                                        data=dictionary_lookup_feats[feature_file],
                                        doc_name=doc_name
                                    )

                            if 'emotion' in features.keys():
                                corpus.update_properties(
                                    feature='emotion',
                                    data=emotion.feat_doc(doc=doc),
                                    doc_name=doc_name
                                )

                            if 'lexical_richness' in features.keys():
                                corpus.update_properties(
                                    feature='lexical_richness',
                                    data=lexical_richness.feat_doc(doc=doc),
                                    doc_name=doc_name
                                )

                            if 'surface' in features.keys():
                                corpus.update_properties(
                                    feature='surface',
                                    data=surface.feat_doc(doc=doc),
                                    doc_name=doc_name
                                )

                            if 'syntax_dependency_metrics' in features.keys() or 'syntax_dependency_tree' in features.keys():
                                syntax_dep_feats = syntax_dependency.feat_doc(doc=doc)
                                for feature_file in syntax_dep_feats.keys():
                                    corpus.update_properties(
                                        feature=feature_file,
                                        data=syntax_dep_feats[feature_file],
                                        doc_name=doc_name
                                    )

                            if 'syntax_constituency_metrics' in features.keys() or 'syntax_constituency_tree' in features.keys():
                                syntax_const_feats = syntax_constituency.feat_doc(doc=doc, plain_text=plain_text)
                                for feature_file in syntax_const_feats.keys():
                                    corpus.update_properties(
                                        feature=feature_file,
                                        data=syntax_const_feats[feature_file],
                                        doc_name=doc_name
                                    )

                            if ('wordnet_synsets' in features.keys() or
                                'wordnet_senses' in features.keys() or
                                'wordnet_semantic_relations' in features.keys()
                            ) and lang in ConfLanguages().wordnet_languages:

                                sem_wordnet_feats = semantics_wordnet.feat_doc(doc=doc)

                                for feature_file in sem_wordnet_feats.keys():
                                    corpus.update_properties(
                                        feature=feature_file,
                                        data=sem_wordnet_feats[feature_file],
                                        doc_name=doc_name
                                    )

                            corpus.update_bleu_props(
                                doc=doc,
                                doc_name=doc_name
                            )

                    if 'corpus_characteristics' in tasks:
                        corpus.macro_features['corpus_characteristics'] = corpus.count_characteristics()
                        write_macro_features(
                            path_summary=path_summary,
                            corpus=corpus,
                            feature_name='corpus_characteristics',
                            file_format_features=file_format_features
                        )

                        path_summary_cc = path_counts + os.sep + 'corpus_characteristics'
                        if not os.path.isdir(path_summary_cc):
                            os.mkdir(path_summary_cc)

                        write_df_to_file(
                            data=pd.DataFrame(corpus.document_cnt_characteristics).transpose(),
                            path_file=path_summary_cc + os.sep + corpus.name,
                            file_format_features=file_format_features
                        )

                    for feat in corpus.features.keys():

                        df_corpus_features = pd.DataFrame(corpus.features[feat]).rename_axis('document')

                        path_feature_bundle = path_features + os.sep + feat
                        if not os.path.isdir(path_feature_bundle):
                            os.mkdir(path_feature_bundle)

                        write_df_to_file(
                            data=pd.DataFrame(df_corpus_features, dtype='float32'),
                            path_file=path_feature_bundle + os.sep + corpus.name + '_' + feat,
                            file_format_features=file_format_features
                        )

                        path_features_summary = path_summary + os.sep + feat
                        if not os.path.isdir(path_features_summary):
                            os.mkdir(path_features_summary)

                        if not df_corpus_features.empty:
                            summary_df = df_corpus_features.describe().transpose()
                        else:
                            summary_df = pd.DataFrame()

                        if feat == 'emotion':
                            corpus.macro_features['emotion'] = emotion.feat_corpus(corpus)
                            write_macro_features(
                                path_summary=path_summary,
                                corpus=corpus,
                                feature_name='emotion',
                                file_format_features=file_format_features
                            )
                            corpus.emotion.clear()

                            summary_df['corpus wise'] = pd.DataFrame(corpus.macro_features['emotion'])

                        elif feat == 'lexical_richness':

                            corpus.macro_features['lexical_richness'] = lexical_richness.feat_corpus(
                                corpus=corpus
                            )
                            write_macro_features(
                                path_summary=path_summary,
                                corpus=corpus,
                                feature_name='lexical_richness',
                                file_format_features=file_format_features
                            )

                            function_words = corpus.lexical_richness['function_words']

                            corpus.lexical_richness.clear()
                            summary_df['corpus wise'] = pd.DataFrame(corpus.macro_features['lexical_richness']).round(2)

                        elif feat == 'surface':

                            corpus.macro_features['surface'] = surface.feat_corpus(corpus)

                            write_macro_features(
                                path_summary=path_summary,
                                corpus=corpus,
                                feature_name='surface',
                                file_format_features=file_format_features
                            )

                            if bool(self.config['settings']['store_sources']) == False:
                                corpus.surface.clear()
                            summary_df['corpus wise'] = pd.DataFrame(corpus.macro_features['surface']).round(2)

                        elif feat == 'syntax_dependency_metrics':

                            corpus.macro_features['syntax_dependency_metrics'] = syntax_dependency.feat_corpus(corpus)

                            write_macro_features(
                                path_summary=path_summary,
                                corpus=corpus,
                                feature_name='syntax_dependency_metrics',
                                file_format_features=file_format_features
                            )

                            if bool(self.config['settings']['store_sources']) == False:
                                corpus.syntax.clear()
                            summary_df['corpus wise'] = pd.DataFrame(corpus.macro_features['syntax_dependency_metrics'])

                        elif feat == 'syntax_constituency_metrics':

                            corpus.macro_features['syntax_constituency_metrics'] = syntax_constituency.feat_corpus(
                                corpus)

                            write_macro_features(
                                path_summary=path_summary,
                                corpus=corpus,
                                feature_name='syntax_constituency_metrics',
                                file_format_features=file_format_features
                            )

                            if bool(self.config['settings']['store_sources']) == False:
                                corpus.syntax.clear()
                            summary_df['corpus wise'] = pd.DataFrame(
                                corpus.macro_features['syntax_constituency_metrics'])


                        elif feat == 'wordnet_semantic_relations':
                            semantics_wordnet_corpus = semantics_wordnet.feat_corpus(corpus)

                            corpus.macro_features['wordnet_semantic_relations'] = semantics_wordnet_corpus
                            write_macro_features(
                                path_summary=path_summary,
                                corpus=corpus,
                                feature_name='wordnet_semantic_relations',
                                file_format_features=file_format_features
                            )
                            summary_df['corpus wise'] = pd.DataFrame(
                                corpus.macro_features['wordnet_semantic_relations'])

                        write_df_to_file(
                            data=summary_df.round(2),
                            path_file=path_features_summary + os.sep + corpus.name + '_' + feat + '_summary',
                            file_format_features=file_format_features
                        )

                    for feat in corpus.counts.keys():
                        logging.info('----')
                        logging.info('counts: ' + feat)

                        path_count_bundle = path_counts + os.sep + feat
                        if not os.path.isdir(path_count_bundle):
                            os.mkdir(path_count_bundle)

                        if 'features' in tasks and feat != 'surface' and feat != 'lexical_richness':

                            logging.info('counts with feature: ' + feat)

                            path_feature_bundle = path_features + os.sep + feat
                            if not os.path.isdir(path_feature_bundle):
                                os.mkdir(path_feature_bundle)

                            corpus_features = {}
                            for item in corpus.counts[feat]:
                                corpus_features[item] = {}

                                for doc_name in corpus.counts[feat][item]:
                                    corpus_features[item][doc_name] = (corpus.counts[feat][item][doc_name] /
                                                                       corpus.document_cnt_characteristics[doc_name][
                                                                           'tokens'])

                            handle_features_output(
                                path_features=path_features,
                                feat=feat,
                                df_corpus_features=pd.DataFrame.from_dict(data=corpus_features,
                                                                          dtype='float32').rename_axis('document'),
                                corpus=corpus,
                                file_format_features=file_format_features,
                                path_summary=path_summary
                            )

                        df_corpus_counts = pd.DataFrame(corpus.counts[feat]).rename_axis('document')
                        corpus.counts[feat].clear()  # clear storage

                        write_df_to_file(
                            data=df_corpus_counts,
                            path_file=path_count_bundle + os.sep + corpus.name + '_corpus_counts_' + feat,
                            file_format_features=file_format_features
                        )

                        sum_all_corpora_counts = {corpus.name: {}}
                        for key, value in zip(list(df_corpus_counts.sum().index), list(df_corpus_counts.sum())):
                            sum_all_corpora_counts[corpus.name][key] = value

                        sum_all_corpora(
                            path_counts=path_counts,
                            sum_all_corpora_counts=sum_all_corpora_counts,
                            feature_name=feat,
                            file_format_features=file_format_features
                        )

                    if 'ngrams' in features.keys():

                        cnt_ngrams_corpus = {}

                        for n in features['ngrams']:

                            tfid_vectorizer = TfidfVectorizer(
                                ngram_range=(n, n),
                                lowercase=False,
                                tokenizer=lambda j: j,
                                max_features=self.config['settings']['most_frequent_words'],
                                stop_words=None
                            )
                            x_tfidf = tfid_vectorizer.fit_transform(corpus.documents.values())

                            df_tfidf = pd.DataFrame(x_tfidf.toarray(),
                                                    columns=list(tfid_vectorizer.get_feature_names_out()))
                            df_tfidf['document'] = list(corpus.documents.keys())
                            df_tfidf = df_tfidf.set_index('document')

                            path_feature_bundle = path_features + os.sep + str(n) + '_ngrams_tfidf'
                            if not os.path.isdir(path_feature_bundle):
                                os.mkdir(path_feature_bundle)

                            path_file = path_feature_bundle + os.sep + corpus.name + '_' + str(n) + '_ngrams_tfidf'

                            write_df_to_file(
                                data=df_tfidf,
                                path_file=path_file,
                                file_format_features=file_format_features
                            )

                    if bool(self.config['settings']['store_sources']):

                        path_storage_sources = self.config['output']['path_sources']

                        path_corpus_res = path_storage_sources + os.sep + corpus.name
                        if not os.path.isdir(path_corpus_res):
                            os.mkdir(path_corpus_res)

                        if 'compare' in self.config.keys():

                            if 'bleu' in self.config['compare']:
                                with open(
                                        file=path_corpus_res + os.sep + corpus.name + '__' + 'bleu_documents' + '.json',
                                        mode='w',
                                        encoding='utf-8'
                                ) as f:
                                    json.dump(corpus.documents, f, ensure_ascii=False)
                                logging.info(
                                    'Source: ' + path_corpus_res + os.sep + corpus.name + '__' + 'bleu_documents' + '.json')

                        for n in corpus.ngrams:
                            with open(
                                    file=path_corpus_res + os.sep + corpus.name + '__' + 'ngrams_' + str(n) + '.json',
                                    mode='w',
                                    encoding='utf-8'
                            ) as f:
                                json.dump(corpus.ngrams[n], f, ensure_ascii=False)
                                corpus.ngrams[n].clear()
                            logging.info('Source: ' + path_corpus_res + os.sep + corpus.name + '__' + 'ngrams_' + str(
                                n) + '.json')

                        for term in corpus.terminologies:
                            with open(
                                    file=path_corpus_res + os.sep + corpus.name + '__' + term + '.json',
                                    mode='w',
                                    encoding='utf-8'
                            ) as f:
                                json.dump(corpus.terminologies[term], f, ensure_ascii=False)
                            logging.info('Source: ' + path_corpus_res + os.sep + corpus.name + '__' + term + '.json')

                        if 'features' in tasks:

                            if 'lexical_richness' in self.config['features'].keys() and 'lexical_richness' in features:
                                with open(
                                        file=path_corpus_res + os.sep + corpus.name + '__' + 'lexical_richness_function_words' + '.json',
                                        mode='w',
                                        encoding='utf-8'
                                ) as f:
                                    json.dump(dict(collections.Counter(function_words)), f, ensure_ascii=False)
                                logging.info(
                                    'Source: ' + path_corpus_res + os.sep + corpus.name + '__' + 'function_words' + '.json')

                            if 'surface' in self.config['features'].keys() and 'surface' in features:

                                for elem in ['syllables', 'letter_tokens', 'no_digit_tokens', 'sentences',
                                             'toks_min_three_syllables', 'toks_larger_six_letters',
                                             'toks_one_syllable']:

                                    if elem in corpus.surface.keys():
                                        with open(
                                                file=path_corpus_res + os.sep + corpus.name + '__' + 'surface_' + elem + '.json',
                                                mode='w',
                                                encoding='utf-8'
                                        ) as f:
                                            json.dump(dict(collections.Counter(corpus.surface[elem])), f,
                                                      ensure_ascii=False)
                                        logging.info(
                                            'Source: ' + path_corpus_res + os.sep + corpus.name + '__' + elem + '.json')
                                corpus.surface.clear()

            logging.info(
                '===============================================================================================')
            logging.info('Processing of Language ' + ConfLanguages().lang_def[lang] + ' done.')
            logging.info(
                '===============================================================================================')

            if 'features' in tasks:
                logging.info('Output of features in ' + str(path_features))
            if 'counts' in tasks and corpora:
                logging.info('Output of counts in ' + str(path_counts))
                logging.info(
                    '===============================================================================================')

        if 'compare' in tasks:
            logging.info(
                '===============================================================================================')
            logging.info('Compare sources.')
            logging.info(
                '===============================================================================================')

            analysis = CompareAnalytics(
                path_sources=self.config['output']['path_sources'],
                corpora=self.config['corpora'],
                path_compare=self.config['output']['path_compare'],
                compare_tasks=self.config['compare'],
                file_format_features=file_format_features,
                most_frequent_words=self.config['settings']['most_frequent_words'],
                tasks=self.config['settings']['tasks']
            )

            analysis.source_analytics()

        if 'features_detail' in tasks:
            logging.info(
                '===============================================================================================')
            logging.info('Feature details.')
            logging.info(
                '===============================================================================================')

            feature_properties = {}
            feature_properties_parent = {}
            corpora = set()

            for feature in self.config['features']:

                for feature_file in glob.glob(self.config['output']['path_features'] + os.sep + '**/', recursive=True):

                    for f in glob.glob(feature_file + os.sep + '*.csv'):

                        feat = f.split(os.sep)[-2]

                        if 'ngrams_tfidf' not in feat and feat == feature:
                            logging.info('Read feature [' + feat + '] file ' + f)
                            data = pd.read_csv(
                                f,
                                index_col=0,
                                keep_default_na=False,
                                low_memory=False
                            ).transpose().to_dict()
                            corp = os.path.basename(f).replace('.csv', '').replace('_' + f.split(os.sep)[-2], '')

                            if corp in self.config['corpora']:
                                corpora.add(corp)
                                for file in data:
                                    for e in data[file]:
                                        adapt_feat = feat + '__' + e
                                        if adapt_feat not in feature_properties.keys():
                                            feature_properties[adapt_feat] = {}
                                            feature_properties_parent[adapt_feat] = feat
                                        if corp not in feature_properties[adapt_feat].keys():
                                            feature_properties[adapt_feat][corp] = []
                                        if not feature_properties[adapt_feat][corp]:
                                            feature_properties[adapt_feat][corp] = [data[file][e]]
                                        else:
                                            temp = feature_properties[adapt_feat][corp]
                                            temp.append(data[file][e])
                                            feature_properties[adapt_feat][corp] = temp

                path_features_detail = self.config['output']['path_features_detail']
                if not os.path.isdir(path_features_detail):
                    os.mkdir(path_features_detail)

                logging.info('Detailed features in: ' + path_features_detail)

                for feat_prop in feature_properties:

                    path_sub_feat_dir = path_features_detail + os.sep + feature_properties_parent[feat_prop]
                    if not os.path.isdir(path_sub_feat_dir):
                        os.mkdir(path_sub_feat_dir)

                    path_sub_feat_file = path_sub_feat_dir + os.sep + feat_prop

                    data = [feature_properties[feat_prop][d] for d in feature_properties[feat_prop]]
                    index_labels = list(feature_properties[feat_prop].keys())

                    for empty_corpus in [corpus for corpus in corpora if
                                         corpus not in feature_properties[feat_prop].keys()]:
                        data.append([])
                        index_labels.append(empty_corpus)

                    write_df_to_file(
                        data=pd.DataFrame(data, index=index_labels),
                        path_file=path_sub_feat_file,
                        file_format_features=file_format_features
                    )

        if 'plot' in tasks and 'features_detail' in tasks:
            logging.info('Tasks: plot')
            logging.info(self.config['output']['path_features_detail'])
            path_features_detail = self.config['output']['path_features_detail']

            for feature_file in glob.glob(path_features_detail + os.sep + '**/', recursive=True):

                for f in glob.glob(feature_file + os.sep + '*.csv'):

                    if feature_file.split(os.sep)[len(feature_file.split(os.sep)) - 2] in self.config[
                        'features'].keys():
                        data = pd.read_csv(f, index_col=0, keep_default_na=False)
                        data_bp = data.transpose().to_dict()

                        boxplot_2dim_array(
                            title="Feature '" + os.path.basename(f).replace('.csv', '') + "'",
                            labels=data.index.values.tolist(),
                            file_plot=f.replace('.csv', ''),
                            data=[[float(data_bp[keys][vals]) for vals in data_bp[keys] if data_bp[keys][vals]] for keys
                                  in data_bp],
                            file_format_plots=file_format_plots,
                            feature_cfg=os.path.basename(f).replace('.csv', ''),
                            height=int(self.config['settings']['boxplot_height'])
                        )

        if 'cluster' in tasks:
            if os.path.isdir(self.config['output']['path_features']):

                logging.info(
                    '===============================================================================================')
                logging.info('Start clustering')
                logging.info('path_features: ' + self.config['output']['path_features'])
                logging.info('path_clusters: ' + self.config['output']['path_clusters'])

                logging.info(
                    '===============================================================================================')

                from dopameter.cluster import ClusterCorpora
                cluster = ClusterCorpora(
                    corpora=self.config['corpora'],
                    features=self.config['features'],
                    path_features=self.config['output']['path_features'],
                    path_clusters=self.config['output']['path_clusters'],
                    feature_file_format_for_clustering=self.config['settings']['file_format_clustering'],
                    diagram_file_formats=file_format_plots,
                    settings=self.config['cluster'],
                    tasks=self.config['settings']['tasks'],
                    file_format_features=file_format_features
                )
                cluster.compute_clusters()
            else:
                exit("The given directory of features " + self.config['output'][
                    'path_features'] + " is not existing. No clusting started!")

        logging.info('===============================================================================================')
        logging.info('Running DoPa Meter done.')
        logging.info('===============================================================================================')
