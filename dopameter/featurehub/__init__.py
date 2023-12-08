import json
import os

import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

from dopameter.configuration.installation import ConfLanguages
from dopameter.analytics.summarization import df_to_file, sum_all_corpora_to_file, macro_features_to_file, features_to_file


def process_feature_hub(config, tasks):
    #global path_features
    logging.info('Start running Feature Hub.')

    path_summary = config['output']['path_summary']
    if not os.path.isdir(path_summary):
        os.mkdir(path_summary)

    if 'counts' in tasks or 'corpus_characteristics' in tasks:
        path_counts = config['output']['path_counts']
        if not os.path.isdir(path_counts):
            os.mkdir(path_counts)

    if 'features' in tasks or 'counts' in tasks:
        features = config['features']
        if not features:
            raise ValueError(
                "Stop running task 'features'! - No features defined! Define 'features' in your config file!")
        else:
            logging.info('features: ' + str(features))

        if 'external_resources' in config.keys():
            external_resources = config['external_resources']
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

        if bool(config['settings']['store_sources']):
            path_storage_sources = config['output']['path_sources']
            if not os.path.isdir(path_storage_sources):
                os.mkdir(path_storage_sources)

        if 'features' in tasks:
            path_features = config['output']['path_features']
            if not os.path.isdir(path_features):
                os.mkdir(path_features)

            if 'ngrams' in features.keys():
                features['ngrams'] = [int(i) for i in features['ngrams']]
    else:
        features = {}
        external_resources = {}

    from dopameter.language_collection.corpora_collection.corpora import SetUpCorpora

    col_by_lang = SetUpCorpora(
        corpora=config['corpora'],
        features=features,
        external_resources=external_resources
    ).create_corpora()

    if len(col_by_lang) > 1:
        logging.info(str(len(col_by_lang)) + ' languages to process: ' + str(
            [ConfLanguages().lang_def[lang] for lang in col_by_lang]))
    elif len(col_by_lang) == 1:
        logging.info(str(len(col_by_lang)) + ' language to process: ' + str(
            [ConfLanguages().lang_def[lang] for lang in col_by_lang]))
    else:
        exit('Routine aborted! There are no languages to process selected. Configure minimal 1 language set-up!')

    collection_names = {col_: [corpus.name for lang in col_by_lang for col in col_by_lang[lang].collections for corpus in col_by_lang[lang].collections[col].corpora if col == col_] for lang in col_by_lang for col_ in col_by_lang[lang].collections}
    if len(collection_names) > 1:
        logging.info(str(len(collection_names)) + ' collections to process: ' + str(list(collection_names.keys())))
    elif len(collection_names) == 1:
        logging.info(str(len(collection_names)) + ' collection to process: ' + str(list(collection_names.keys())))
    else:
        exit('Routine aborted! There are no corpora selected. Configure minimal 1 text corpus with files!')

    # languages and their collections {col_by_lang[lang].name: list(col_by_lang[lang].collections.keys()) for lang in col_by_lang}

    corpora_names = [corpus.name for lang in col_by_lang for col in col_by_lang[lang].collections for corpus in col_by_lang[lang].collections[col].corpora]
    if len(corpora_names) > 1:
        logging.info(str(len(corpora_names)) + ' corpora to process: ' + str(corpora_names))
    elif len(corpora_names) == 1:
        logging.info(str(len(corpora_names)) + ' corpus to process: ' + str(corpora_names))
    else:
        exit('Routine aborted! There are no corpora selected. Configure minimal 1 text corpus with files!')

    from dopameter.configuration.pipeline import PreProcessingPipline

    pipeline = PreProcessingPipline(config=config)

    corp_i = 0
    for lang in col_by_lang:
        logging.info('===============================================================================================')
        logging.info("Start to process language '" + ConfLanguages().lang_def[lang] + "'")
        logging.info('===============================================================================================')
        nlp = pipeline.create_nlp(lang)

        from dopameter.language_collection.corpora_collection.corpora.corpus.document import BasicCharacteristics

        basic_chars = BasicCharacteristics()

        if 'token_characteristics' in features.keys():
            from dopameter.featurehub.token_characteristics import TokenCharacteristics

            tok_chars = TokenCharacteristics(
                features=features['token_characteristics'],
            )

        if 'pos' in features.keys():
            from dopameter.featurehub.pos import POSFeatures

            pos = POSFeatures(
                nlp=nlp,
                features=features['pos']
            )

        if 'ner' in features.keys():
            from dopameter.featurehub.ner import NERFeatures

            ner = NERFeatures(
                nlp=nlp,
                features=features['ner']
            )

        if 'ngrams' in features.keys():
            from dopameter.featurehub.ngrams import NGramFeatures

            ngrams = NGramFeatures(
                config=config,
                features=features['ngrams']
            )
            features['ngrams'] = [int(i) for i in features['ngrams']]

        if 'lexical_diversity' in features.keys():
            from dopameter.featurehub.lexical_diversity import LexicalDiversityFeatures

            lexical_diversity = LexicalDiversityFeatures(
                features=features['lexical_diversity']
            )

        if 'surface' in features.keys():
            import spacy_syllables
            nlp.add_pipe("syllables", after="tagger")  # 'import spacy_syllables' is used before!
            if lang == 'de':
                from dopameter.featurehub.surface.de import SurfaceFeaturizesDE
                surface = SurfaceFeaturizesDE(features=features['surface'])
            elif lang == 'en':
                from dopameter.featurehub.surface.en import SurfaceFeaturizesEN
                surface = SurfaceFeaturizesEN(features=features['surface'])
            else:
                from dopameter.featurehub.surface import SurfaceFeaturizes
                surface = SurfaceFeaturizes(features=features['surface'])

        if set(features.keys()).intersection({'syntax_dependency_metrics', 'syntax_dependency_tree'}):
            from dopameter.featurehub.syntax.dependency import DependencyFeatures

            syntax_dependency = DependencyFeatures(
                features=features
            )

        if set(features.keys()).intersection({'syntax_constituency_metrics', 'syntax_constituency_tree'}):
            from dopameter.featurehub.syntax.constituency import ConstituencyFeatures

            syntax_constituency = ConstituencyFeatures(
                lang=lang,
                features=features
            )

        if set(features.keys()).intersection({'wordnet_synsets', 'wordnet_senses', 'wordnet_semantic_relations'}):
            from dopameter.featurehub.semantics.wordnet import WordNetFeatures

            semantics_wordnet = WordNetFeatures(
                lang=lang,
                features=features
            )

        if 'dictionary_lookup' in features.keys():
            from dopameter.featurehub.semantics.dictionary_lookup.simple import DictionaryLookUp

            if 'dictionary_lookup' in features.keys():
                dictionary_lookup = DictionaryLookUp(
                    nlp=nlp,
                    path_dictionaries=config['external_resources']['dictionaries'],
                    file_format_dicts=config['settings']['file_format_dicts'],
                    features=features['dictionary_lookup']
                )

        if 'emotion' in features.keys():
            from dopameter.featurehub.emotion import EmotionFeatures

            emotion = EmotionFeatures(
                lang=lang,
                features=features['emotion']
            )

        logging.info('===============================================================================================')
        logging.info('Finished initialization of language ' + ConfLanguages().lang_def[lang])
        logging.info('===============================================================================================')

        for collect in col_by_lang[lang].collections:

            logging.info('-----------------------------------------------------------------------------------------------')
            logging.info('\t\tStart to process collection ' + collect + '.')
            logging.info('-----------------------------------------------------------------------------------------------')

            collection = col_by_lang[lang].collections[collect]

            for corpus in col_by_lang[lang].collections[collect].corpora:

                corp_i += 1
                logging.info('-----------------------------------------------------------------------------------------------')
                logging.info('\tStart to process corpus (' + str(corp_i) + '/' + str(len(corpora_names)) + '): ' + corpus.name)
                logging.info('\t# ' + str(len(corpus.files)) + ' files from ' + corpus.path)
                logging.info('-----------------------------------------------------------------------------------------------')

                for i, f in enumerate(corpus.files):
                    logging.info("process corpus '" + corpus.name + "' (" + str(corp_i) + '/' + str(len(corpora_names)) + ') - file (' + str(i + 1) + '/' + str(len(corpus.files)) + ') ' + str(f))

                    doc_name = os.path.basename(f)
                    plain_text = open(file=f, encoding=corpus.encoding).read()
                    doc = nlp(plain_text)

                    if doc._.n_tokens == 0:
                        logging.warning('File ' + f + ' has 0 tokens. The document is not processed!')

                    elif doc._.n_tokens > 0:

                        corpus.update_properties(
                            feature='basic_counts',
                            data=basic_chars.count_doc(doc=doc),
                            doc_name=doc_name
                        )

                        if 'token_characteristics' in features.keys():
                            corpus.update_properties(
                                feature='token_characteristics',
                                data=tok_chars.feat_doc(doc=doc),
                                doc_name=doc_name
                            )

                        if 'pos' in features.keys():
                            corpus.update_properties(
                                feature='pos',
                                data=pos.feat_doc(doc=doc),
                                doc_name=doc_name
                            )

                        if 'ner' in features.keys():
                            corpus.update_properties(
                                feature='ner',
                                data=ner.feat_doc(doc=doc),
                                doc_name=doc_name
                            )

                        if 'ngrams' in features.keys():
                            corpus.update_properties(
                                feature='ngrams',
                                data=ngrams.feat_doc(doc=doc),
                                doc_name=doc_name
                            )

                        if 'lexical_diversity' in features.keys():
                            corpus.update_properties(
                                feature='lexical_diversity',
                                data=lexical_diversity.feat_doc(doc=doc),
                                doc_name=doc_name
                            )

                        if 'surface' in features.keys():
                            corpus.update_properties(
                                feature='surface',
                                data=surface.feat_doc(doc=doc),
                                doc_name=doc_name
                            )

                        if set(features.keys()).intersection({'syntax_dependency_metrics', 'syntax_dependency_metrics'}):
                            syntax_dep_feats = syntax_dependency.feat_doc(doc=doc)
                            for feature_file in syntax_dep_feats.keys():
                                corpus.update_properties(
                                    feature=feature_file,
                                    data=syntax_dep_feats[feature_file],
                                    doc_name=doc_name
                                )

                        if set(features.keys()).intersection({'syntax_constituency_metrics', 'syntax_constituency_tree'}):
                            syntax_const_feats = syntax_constituency.feat_doc(doc=doc, plain_text=plain_text)
                            for feature_file in syntax_const_feats.keys():
                                corpus.update_properties(
                                    feature=feature_file,
                                    data=syntax_const_feats[feature_file],
                                    doc_name=doc_name
                                )

                        if set(features.keys()).intersection({'wordnet_synsets', 'wordnet_senses', 'wordnet_semantic_relations'}) and lang in ConfLanguages().wordnet_languages:
                            sem_wordnet_feats = semantics_wordnet.feat_doc(doc=doc)

                            for feature_file in sem_wordnet_feats.keys():
                                corpus.update_properties(
                                    feature=feature_file,
                                    data=sem_wordnet_feats[feature_file],
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

                        corpus.update_documents(
                            doc=doc,
                            doc_name=doc_name
                        )

                if 'corpus_characteristics' in tasks:
                    corpus.macro_features['corpus_characteristics'] = corpus.count_characteristics()
                    macro_features_to_file(
                        path_summary=path_summary,
                        corpus=corpus,
                        feature_name='corpus_characteristics',
                        file_format_features=config['settings']['file_format_features']
                    )

                    path_summary_cc = path_counts + os.sep + 'corpus_characteristics'
                    if not os.path.isdir(path_summary_cc):
                        os.mkdir(path_summary_cc)

                    df_to_file(
                        data=pd.DataFrame(data=corpus.document_cnt_characteristics).transpose(),
                        path_file=path_summary_cc + os.sep + corpus.name,
                        file_format_features=config['settings']['file_format_features']
                    )

                collection.sizes.update_sizes_by_corpus_scores(corpus.sizes)

                for feat in corpus.features.keys():

                    df_corpus_features = pd.DataFrame(data=corpus.features[feat]).rename_axis('document')

                    path_feature_bundle = path_features + os.sep + feat
                    if not os.path.isdir(path_feature_bundle):
                        os.mkdir(path_feature_bundle)

                    if feat != 'ngrams':

                        df_to_file(
                            data=pd.DataFrame(data=df_corpus_features, dtype='float32'),
                            path_file=path_feature_bundle + os.sep + corpus.name + '_' + feat,
                            file_format_features=config['settings']['file_format_features']
                        )

                        path_features_summary = path_summary + os.sep + 'summary_documents'
                        if not os.path.isdir(path_features_summary):
                            os.mkdir(path_features_summary)

                        path_features_summary = path_features_summary + os.sep + feat
                        if not os.path.isdir(path_features_summary):
                            os.mkdir(path_features_summary)

                        if not df_corpus_features.empty:
                            summary_df = df_corpus_features.describe().transpose()
                        else:
                            summary_df = pd.DataFrame()

                    if feat == 'lexical_diversity':
                        corpus.macro_features['lexical_diversity'] = lexical_diversity.feat_corpus(corpus=corpus)

                    elif feat == 'surface':
                        corpus.macro_features['surface'] = surface.feat_corpus(corpus=corpus)

                    elif feat == 'syntax_dependency_metrics':
                        corpus.macro_features['syntax_dependency_metrics'] = syntax_dependency.feat_corpus(corpus=corpus)

                    elif feat == 'syntax_constituency_metrics':
                        corpus.macro_features['syntax_constituency_metrics'] = syntax_constituency.feat_corpus(corpus=corpus)

                    elif feat == 'wordnet_semantic_relations':
                        corpus.macro_features['wordnet_semantic_relations'] = semantics_wordnet.feat_corpus(corpus=corpus)

                    elif feat == 'emotion':
                        corpus.macro_features['emotion'] = emotion.feat_corpus(corpus=corpus)

                    collection.update_scores(feature=feat, resources=corpus.resources)

                    if corpus.features[feat] != {}:
                        macro_features_to_file(
                            path_summary=path_summary,
                            corpus=corpus,
                            feature_name=feat,
                            file_format_features=config['settings']['file_format_features']
                        )
                        summary_df['corpus wise'] = pd.DataFrame(data=corpus.macro_features[feat])

                        df_to_file(
                            data=summary_df.round(4),
                            path_file=path_features_summary + os.sep + corpus.name + '_' + feat + '_summary',
                            file_format_features=config['settings']['file_format_features']
                        )

                for feat in corpus.counts.keys():

                    if 'counts' in tasks:
                        logging.info('----')
                        logging.info('counts: ' + feat)

                        path_count_bundle = path_counts + os.sep + 'summary_documents'
                        if not os.path.isdir(path_count_bundle):
                            os.mkdir(path_count_bundle)
                        path_count_bundle = path_count_bundle + os.sep + feat
                        if not os.path.isdir(path_count_bundle):
                            os.mkdir(path_count_bundle)

                        df_corpus_counts = pd.DataFrame(data=corpus.counts[feat]).rename_axis('document')

                        df_to_file(
                            data=df_corpus_counts,
                            path_file=path_count_bundle + os.sep + corpus.name + '_corpus_counts_' + feat,
                            file_format_features=config['settings']['file_format_features']
                        )

                        sum_all_corpora_counts = {corpus.name: {}}
                        avg_all_corpora_counts = {corpus.name: {}}
                        for key, value in zip(list(df_corpus_counts.sum().index), list(df_corpus_counts.sum())):
                            sum_all_corpora_counts[corpus.name][key] = value
                            avg_all_corpora_counts[corpus.name][key] = (value / corpus.sizes.tokens_cnt)

                        corpus.macro_features[feat] = {'features': avg_all_corpora_counts[corpus.name]}

                    if 'features' in tasks and feat != 'surface' and feat != 'lexical_diversity' :

                        logging.info('counts with feature: ' + feat)

                        path_feature_bundle = path_features + os.sep + feat
                        if not os.path.isdir(path_feature_bundle):
                            os.mkdir(path_feature_bundle)

                        corpus_features = {}
                        for item in corpus.counts[feat]:
                            corpus_features[item] = {}

                            for doc_name in corpus.counts[feat][item]:
                                corpus_features[item][doc_name] = (corpus.counts[feat][item][doc_name] / corpus.document_cnt_characteristics[doc_name]['tokens'])

                        features_to_file(
                            path_features=path_features,
                            feat=feat,
                            df_corpus_features=pd.DataFrame.from_dict(
                                data=corpus_features,
                                dtype='float32').rename_axis('document'),
                            corpus=corpus,
                            file_format_features=config['settings']['file_format_features']
                        )

                        summary_df = pd.DataFrame.from_dict(
                            data=corpus_features,
                            dtype='float32'
                        ).rename_axis('document').describe().transpose()

                        if 'counts' in tasks:
                            summary_df['corpus wise'] = pd.DataFrame(data=avg_all_corpora_counts)

                        path_features_summary = path_summary + os.sep + 'summary_documents'
                        if not os.path.isdir(path_features_summary):
                            os.mkdir(path_features_summary)

                        path_features_summary = path_features_summary + os.sep + feat
                        if not os.path.isdir(path_features_summary):
                            os.mkdir(path_features_summary)

                        if not os.path.isdir(path_features_summary):
                            os.mkdir(path_features_summary)

                        df_to_file(
                            data=summary_df.round(4),
                            path_file=path_features_summary + os.sep + corpus.name + '_' + feat + '_summary',
                            file_format_features=config['settings']['file_format_features']
                        )

                        macro_features_to_file(
                            path_summary=path_summary,
                            corpus=corpus,
                            feature_name=feat,
                            file_format_features=config['settings']['file_format_features']
                        )

                        collection.update_counts(feature=feat, scores=sum_all_corpora_counts[corpus.name])

                        sum_all_corpora_to_file(
                            path_counts=path_counts,
                            sum_all_corpora_counts=sum_all_corpora_counts,
                            feature_name=feat,
                            file_format_features=config['settings']['file_format_features'],
                            corpus_name = corpus.name,
                            collection=corpus.collection_name,
                            language=ConfLanguages().lang_def[lang]
                        )

                if 'ngrams' in features.keys():

                    for n in features['ngrams']:

                        tfid_vectorizer = TfidfVectorizer(
                            ngram_range=(n, n),
                            lowercase=False,
                            tokenizer=lambda j: j,
                            max_features=config['settings']['most_frequent_words'],
                            stop_words=None
                        )

                        if 'mode_ngrams' in config['settings'].keys() and config['settings']['mode_ngrams'] == 'pos':
                            x_tfidf = tfid_vectorizer.fit_transform(raw_documents=corpus.documents_pos.values())
                            n_gram_def = '_ngrams_tfidf_pos'
                        else:
                            x_tfidf = tfid_vectorizer.fit_transform(raw_documents=corpus.documents.values())
                            n_gram_def = '_ngrams_tfidf'

                        df_tfidf = pd.DataFrame(
                            data=x_tfidf.toarray(),
                            columns=list(tfid_vectorizer.get_feature_names_out())
                        )
                        df_tfidf['document'] = list(corpus.documents.keys())
                        df_tfidf = df_tfidf.set_index('document')

                        path_feature_bundle = path_features + os.sep + str(n) + n_gram_def
                        if not os.path.isdir(path_feature_bundle):
                            os.mkdir(path_feature_bundle)

                        df_to_file(
                            data=df_tfidf,
                            path_file=path_feature_bundle + os.sep + corpus.name + '_' + str(n) + n_gram_def,
                            file_format_features=config['settings']['file_format_features']
                        )

                if bool(config['settings']['store_sources']):

                    path_storage_sources = config['output']['path_sources']

                    path_corpus_res = path_storage_sources + os.sep + corpus.name
                    if not os.path.isdir(path_corpus_res):
                        os.mkdir(path_corpus_res)

                    if 'compare' in config.keys():

                        if 'bleu' in config['compare']:
                            with open(
                                    file=path_corpus_res + os.sep + corpus.name + '__' + 'bleu_documents' + '.json',
                                    mode='w',
                                    encoding='utf-8'
                            ) as f:
                                json.dump(
                                    obj=corpus.documents,
                                    fp=f,
                                    ensure_ascii=False
                                )
                            logging.info('Source: ' + path_corpus_res + os.sep + corpus.name + '__' + 'bleu_documents' + '.json')

                    for n in corpus.resources.ngrams:

                        if 'mode_ngrams' in config['settings'].keys() and config['settings']['mode_ngrams'] == 'pos':
                            ngram_file = path_corpus_res + os.sep + corpus.name + '__' + 'ngrams_pos_' + str(n) + '.json'
                        else:
                            ngram_file = path_corpus_res + os.sep + corpus.name + '__' + 'ngrams_' + str(n) + '.json'

                        with open(
                                file=ngram_file,
                                mode='w',
                                encoding='utf-8'
                        ) as f:
                            json.dump(
                                obj=corpus.resources.ngrams[n],
                                fp=f,
                                ensure_ascii=False
                            )

                        logging.info('Source: ' + ngram_file)

                    for term in corpus.resources.terminologies:
                        with open(
                                file=path_corpus_res + os.sep + corpus.name + '__' + term + '.json',
                                mode='w',
                                encoding='utf-8'
                        ) as f:
                            json.dump(
                                obj=corpus.resources.terminologies[term],
                                fp=f,
                                ensure_ascii=False
                            )
                        logging.info('Source: ' + path_corpus_res + os.sep + corpus.name + '__' + term + '.json')

                corpus.clear()

            col_by_lang[lang].sizes.update_sizes_by_corpus_scores(collection.sizes)

            for feat in collection.counts:
                collection.macro_features[feat] = {'features': {key: collection.counts[feat][key] / collection.sizes.tokens_cnt for key in collection.counts[feat]}}
                col_by_lang[lang].update_counts(feature=feat, scores=collection.counts[feat])

                sum_all_corpora_to_file(
                    path_counts=path_counts,
                    sum_all_corpora_counts={collection.name:collection.counts[feat]},
                    feature_name=feat,
                    file_format_features=config['settings']['file_format_features'],
                    corpus_name=[collection_names[collection.name]],
                    collection=collection.name,
                    language=ConfLanguages().lang_def[lang]
                )

            for feat in features:
                if feat == 'lexical_diversity':
                    collection.macro_features[feat] = lexical_diversity.feat_corpus(collection)

                elif feat == 'surface':
                    collection.macro_features[feat] = surface.feat_corpus(collection)

                elif feat == 'syntax_dependency_metrics':
                    collection.macro_features[feat] = syntax_dependency.feat_corpus(collection)

                elif feat == 'wordnet_semantic_relations':
                    collection.macro_features[feat] = semantics_wordnet.feat_corpus(collection)

                elif feat == 'emotion':
                    collection.macro_features[feat] = emotion.feat_corpus(collection)

                col_by_lang[lang].update_scores(feature=feat, resources=collection.resources)
                macro_features_to_file(
                    path_summary=path_summary,
                    corpus=collection,
                    feature_name=feat,
                    file_format_features=config['settings']['file_format_features']
                )
            collection.clear()

            logging.info('-----------------------------------------------------------------------------------------------')
            logging.info("\tProcessing of COLLECTION '" + collect + "' done.")
            logging.info('-----------------------------------------------------------------------------------------------')

        for feat in col_by_lang[lang].counts:
            col_by_lang[lang].macro_features[feat] = {'features': {key: col_by_lang[lang].counts[feat][key] / col_by_lang[lang].sizes.tokens_cnt for key in col_by_lang[lang].counts[feat]}}
            col_by_lang[lang].update_counts(feature=feat, scores=col_by_lang[lang].counts[feat])

            sum_all_corpora_to_file(
                path_counts=path_counts,
                sum_all_corpora_counts={col_by_lang[lang].name: col_by_lang[lang].counts[feat]},
                feature_name=feat,
                file_format_features=config['settings']['file_format_features'],
                corpus_name=[corpus.name for col in col_by_lang[lang].collections for corpus in col_by_lang[lang].collections[col].corpora],
                collection=list(col_by_lang[lang].collections.keys()),
                language=ConfLanguages().lang_def[lang]
            )

        for feat in features:
            if feat == 'lexical_diversity':
                col_by_lang[lang].macro_features[feat] = lexical_diversity.feat_corpus(col_by_lang[lang])

            elif feat == 'surface':
                col_by_lang[lang].macro_features[feat] = surface.feat_corpus(col_by_lang[lang])

            elif feat == 'syntax_dependency_metrics':
                col_by_lang[lang].macro_features[feat] = syntax_dependency.feat_corpus(col_by_lang[lang])

            elif feat == 'wordnet_semantic_relations':
                col_by_lang[lang].macro_features[feat] = semantics_wordnet.feat_corpus(col_by_lang[lang])

            elif feat == 'emotion':
                col_by_lang[lang].macro_features[feat] = emotion.feat_corpus(col_by_lang[lang])

            macro_features_to_file(
                path_summary=path_summary,
                corpus=col_by_lang[lang],
                feature_name=feat,
                file_format_features=config['settings']['file_format_features']
            )

    col_by_lang[lang].clear()

    logging.info('===============================================================================================')
    logging.info("Processing of LANGUAGE '" + ConfLanguages().lang_def[lang] + "' done.")
    logging.info('===============================================================================================')

    if 'features' in tasks:
        logging.info('Output of features in ' + str(path_features))
    if 'counts' in tasks:
        logging.info('Output of counts in ' + str(path_counts))
        logging.info('===============================================================================================')
