import collections
import glob
import json
import os
import nltk
import numpy as np
import pandas as pd
import logging
from dopameter.analytics.summarization import df_to_file
from dopameter.analytics.vis_utils import heatmap_comparison
import dopameter.delta as delta


class CompareAnalytics:
    """Comparison mode

    Parameters
    ----------
    path_sources : str
    corpora : dict
    path_compare : str

    file_format_features : list of str
    most_frequent_words : int
    tasks : list of str

    Notes
    -----

    * **Mandatory:**
      1. Run of `'features'` before or in same run
      2. Store resources (interim results from features) during running `'features'`
        * Configuration in settings:
          * `"store_sources": true`
          * `"path_compare": "path/sources"`
        * [Example sources / interim results of one corpus](../res/results/sources/wiki)

    * **Decentralized approach:**
      * You can run the features from the task `'features'` on different places and store the interim results via `"store_sources": true` in the configuration.
      * If you want to run the compare mode of corpora from different places, copy the path with the interim results from _different_ `"path_compare"` into one directory `"path_compare"` and define it into one [configuration file](../res/example_configurations).

    * It computes intersections and differences of corpora (csv files) and visualize it in heatmaps (`plot mode):
      * Intersection of items, of the frequency of item and both portion-wise
      * Difference of items, of the frequency of item and both portion-wise, example:
      * Intersections and differences stored in json-files.
    * Metrics from language generation and language translation (used from NLTK):
      * [BLEU](https://www.nltk.org/_modules/nltk/translate/bleu_score.html)
      * [METEOR](https://www.nltk.org/howto/meteor.html)
      * [NIST](https://www.nltk.org/_modules/nltk/translate/nist_score.html)
        * Distances metrics (only via vocabulary / $1$-grams):
          * _Burrows' Delta_
          * _Manhattan Distance_
          * _Euclidean Distance_
          * _Squared Euclidean Distance_
          * _Cosine Distance_
          * _Canberra Distance_
          * _Bray-Curtis Distance_
          * _Correlation Distance_
          * _Chebyshev Distance_
          * _Quadratic Delta_
          * _Eder's Delta_
          * _Cosine Delta_
        * The implementation is derived from [PyDelta](https://github.com/cophi-wue/pydelta).


    * Configuration
        "compare": [
          "portion_intersection",
          "portion_intersection_all_occurrences",
          "portion_difference",
          "portion_difference_all_occurrences",
          "bleu",
          "meteor",
          "nist",
          "burrows",
          "manhattan",
          "euclidean",
          "sqeuclidean",
          "cosine",
          "canberra",
          "braycurtis",
          "correlation",
          "chebyshev",
          "quadratic",
          "eder",
          "cosine_delta"
        ]


    """

    def __init__(
        self,
        path_sources,
        corpora,
        path_compare,
        compare_tasks=[
            "intersection",
            "intersection_all_occurrences",
            "intersection_data",
            "portion_intersection",
            "portion_intersection_all_occurrences",
            "difference",
            "difference_all_occurrences",
            "difference_data",
            "portion_difference",
            "portion_difference_all_occurrences"
          ],
        file_format_features=['csv'],
        most_frequent_words=2000,
        tasks=["compare", "plots"],
    ):

        self.path_sources = path_sources
        self.corpora = corpora
        self.path_compare = path_compare
        self.compare_tasks = compare_tasks

        self.file_format_features = file_format_features

        self.most_frequent_words=most_frequent_words
        self.tasks=tasks

        if not os.path.isdir(self.path_compare):
            os.mkdir(self.path_compare)

    def source_analytics_corp_col(self, features):

        logging.info('tasks: compare')

        for feat in features.keys():

            logging.info('compare feature: ' + feat)

            if feat != 'bleu':

                results = {
                    'intersection': {},
                    'intersection_all_occurrences': {},
                    'intersection_data': {},
                    'portion_intersection': {},
                    'portion_intersection_all_occurrences': {},

                    'difference': {},
                    'difference_all_occurrences': {},
                    'difference_data': {},
                    'portion_difference': {},
                    'portion_difference_all_occurrences': {}
                }

                for c_i in sorted(features[feat].keys()):

                    results['intersection'][c_i] = {}
                    results['intersection_all_occurrences'][c_i] = {}
                    results['intersection_data'][c_i] = {}

                    results['portion_intersection'][c_i] = {}
                    results['portion_intersection_all_occurrences'][c_i] = {}

                    results['difference'][c_i] = {}
                    results['difference_all_occurrences'][c_i] = {}
                    results['difference_data'][c_i] = {}

                    results['portion_difference'][c_i] = {}
                    results['portion_difference_all_occurrences'][c_i] = {}

                    for c_j in sorted(features[feat].keys()):

                        if len(features[feat][c_i].keys()) != 0:

                            if 'intersection' in self.compare_tasks:
                                intersection = {x: features[feat][c_i][x] for x in features[feat][c_i].keys() if x in features[feat][c_j].keys()}

                                results['intersection'][c_i][c_j] = len(intersection)
                                results['intersection_all_occurrences'][c_i][c_j] = sum(intersection.values())
                                results['intersection_data'][c_i][c_j] = intersection

                                # portions
                                results['portion_intersection'][c_i][c_j] = round(len(intersection) / len(features[feat][c_i].keys()), 2)
                                results['portion_intersection_all_occurrences'][c_i][c_j] = round(results['intersection_all_occurrences'][c_i][c_j] / len(features[feat][c_i].keys()), 2)

                            if 'differences' in self.compare_tasks:
                                difference = {x: features[feat][c_i][x] for x in features[feat][c_i].keys() if x not in features[feat][c_j].keys()}
                                results['difference'][c_i][c_j] = len(difference)
                                results['difference_all_occurrences'][c_i][c_j] = sum(difference.values())
                                results['difference_data'][c_i][c_j] = difference

                                # difference portions
                                results['portion_difference'][c_i][c_j] = round(len(difference) / len(features[feat][c_i].keys()), 2)
                                results['portion_difference_all_occurrences'][c_i][c_j] = round(results['difference_all_occurrences'][c_i][c_j] / len(features[feat][c_i].keys()), 2)
                        else:
                            logging.info('Content of feature ' + feat + ' is emtpy!')

                for r in results.keys():

                    if 'data' not in r:

                        if results[r] != {} or results[r].values():

                            if r in self.compare_tasks:

                                data = pd.DataFrame(results[r])

                                if not data.empty:

                                    path_compare_detail = self.path_compare + os.sep + feat + os.sep
                                    if not os.path.isdir(path_compare_detail):
                                        os.mkdir(path_compare_detail)

                                    if 'plots' in self.tasks:
                                        heatmap_comparison(
                                            data=pd.DataFrame(results[r]),
                                            title=feat + ' ' + r.replace('_', ' '),
                                            path_plot=path_compare_detail + feat + '_' + r
                                        )

                                    df_to_file(
                                        data=pd.DataFrame(results[r]),
                                        path_file=path_compare_detail + feat + '_' + r,
                                        file_format_features=self.file_format_features
                                    )
                                else:
                                    logging(r + ': data is empty')

                    else:
                        if 'intersection_data' == r and 'intersection' in self.compare_tasks:
                            for key in results['intersection_data']:
                                for val in results['intersection_data'][key]:
                                    if key == val:
                                        results['intersection_data'][key][val] = {}

                            with open(path_compare_detail + os.sep + feat + '_intersection' + '.json', 'w', encoding='utf-8') as f:
                                json.dump(results['intersection_data'], f, ensure_ascii=False, indent=2)

                        if 'difference_data' == r and 'difference' in self.compare_tasks:
                             with open(path_compare_detail + os.sep + feat + '_difference' + '.json', 'w', encoding='utf-8') as f:
                                json.dump(results['difference_data'], f, ensure_ascii=False, indent=2)

            #else:
            if 'bleu' in self.compare_tasks:

                if 'bleu' in self.compare_tasks:
                    logging.info('Computing Bleu-Scores.')
                    bleu_score = {}
                    bleu_map = {}

                if 'meteor' in self.compare_tasks:
                    logging.info('Computing METEOR-Scores.')
                    meteor_score = {}
                    meteor_map = {}

                if 'nist' in self.compare_tasks:
                    logging.info('Computing NIST-Scores.')
                    nist_score = {}
                    nist_map = {}

                for corpus in sorted(features[feat]):

                    logging.info('Init scores.')

                    if 'bleu' in self.compare_tasks:
                        bleu_score[corpus] = {}
                        bleu_map[corpus] = {}

                    if 'meteor' in self.compare_tasks:
                        meteor_score[corpus] = {}
                        meteor_map[corpus] = {}

                    if 'nist' in self.compare_tasks:
                        nist_score[corpus] = {}
                        nist_map[corpus] = {}

                    for cor in sorted(features[feat]):

                        logging.info('Init scores.')

                        if 'bleu' in self.compare_tasks:
                            bleu_score[corpus][cor] = {}

                        if 'meteor' in self.compare_tasks:
                            meteor_score[corpus][cor] = {}

                        if 'nist' in self.compare_tasks:
                            nist_score[corpus][cor] = {}

                        if corpus != cor:

                            logging.info("Creating Matrix-Entries between corpora '" + corpus + "' and '" + cor + "'.")

                            for doc in features[feat][cor]:

                                if 'bleu' in self.compare_tasks:

                                    bleu_score[corpus][cor][doc] = nltk.translate.bleu_score.corpus_bleu(
                                        list_of_references=[features[feat][corpus]],
                                        hypotheses=[features[feat][cor][doc]]
                                    )

                                    #bleu_score[corpus][cor][doc] = nltk.translate.bleu_score.corpus_bleu(
                                    #    list_of_references=[features[feat][corpus].values()],
                                    #    hypotheses=[features[feat][cor][doc]]
                                    #)

                                if 'meteor' in self.compare_tasks:
                                    meteor_score[corpus][cor][doc] = nltk.translate.meteor(
                                        references=list(features[feat][corpus].values()),
                                        hypothesis=features[feat][cor][doc]
                                    )

                                if 'nist' in self.compare_tasks:
                                    try:
                                        nist_score[corpus][cor][doc] = nltk.translate.nist(
                                            references=list(features[feat][corpus].values()),
                                            hypothesis=features[feat][cor][doc]
                                        )
                                    except:
                                        logging.info("Nist_score not computable for document: " + doc + " '" + corpus + "' and '" + cor + "'")

                            if 'bleu' in self.compare_tasks:
                                bleu_map[corpus][cor]   = np.mean(list(bleu_score[corpus][cor].values()))
                            if 'meteor' in self.compare_tasks:
                                meteor_map[corpus][cor] = np.mean(list(meteor_score[corpus][cor].values()))
                            if 'nist' in self.compare_tasks:
                                nist_map[corpus][cor]   = np.mean(list(nist_score[corpus][cor].values()))

                        else:
                            if 'bleu' in self.compare_tasks:
                                bleu_map[corpus][cor] = None
                            if 'meteor' in self.compare_tasks:
                                meteor_map[corpus][cor] = None
                            if 'nist' in self.compare_tasks:
                                nist_map[corpus][cor] = None

                if 'bleu' in self.compare_tasks:

                    path_compare_detail = self.path_compare + os.sep + 'bleu' + os.sep
                    if not os.path.isdir(path_compare_detail):
                        os.mkdir(path_compare_detail)

                    if 'plot' in self.tasks:
                        heatmap_comparison(
                            data=pd.DataFrame(bleu_map),
                            title='bleu scores',
                            path_plot=path_compare_detail + 'bleu_scores'
                        )

                    df_to_file(
                        data=pd.DataFrame(bleu_map),
                        path_file=path_compare_detail + 'bleu_scores',
                        file_format_features=self.file_format_features
                    )

                if 'meteor' in self.compare_tasks:

                    path_compare_detail = self.path_compare + os.sep + 'meteor' + os.sep
                    if not os.path.isdir(path_compare_detail):
                        os.mkdir(path_compare_detail)

                    if 'plot' in self.tasks:
                        heatmap_comparison(
                            data=pd.DataFrame(meteor_map),
                            title='meteor scores',
                            path_plot=path_compare_detail + 'meteor_scores'
                        )

                    df_to_file(
                        data=pd.DataFrame(meteor_map),
                        path_file=path_compare_detail + 'meteor_scores',
                        file_format_features=self.file_format_features
                    )

                if 'nist' in self.compare_tasks:

                    path_compare_detail = self.path_compare + os.sep + 'nist' + os.sep
                    if not os.path.isdir(path_compare_detail):
                        os.mkdir(path_compare_detail)

                    if 'plot' in self.tasks:
                        heatmap_comparison(
                            data=pd.DataFrame(nist_map),
                            title='nist scores',
                            path_plot=path_compare_detail + 'nist_scores'
                        )

                    df_to_file(
                        data=pd.DataFrame(nist_map),
                        path_file=path_compare_detail + 'nist_scores',
                        file_format_features=self.file_format_features
                    )

            if feat == 'ngrams_1':

                path_compare_detail = self.path_compare + os.sep + feat + os.sep
                if not os.path.isdir(path_compare_detail):
                    os.mkdir(path_compare_detail)

                path_compare_distances = self.path_compare + os.sep + 'distances' + os.sep
                if not os.path.isdir(path_compare_distances):
                    os.mkdir(path_compare_distances)

                df = pd.DataFrame(features[feat]).transpose().fillna(0)

                whole_corpus = delta.corpus.Corpus(df)

                corpora = whole_corpus.top_n(int(self.most_frequent_words))

                if 'linear' in self.compare_tasks:
                    linear = delta.functions.linear(corpora)
                    if 'plot' in self.tasks:
                        heatmap_comparison(
                            data=pd.DataFrame(linear),
                            title='linear distance',
                            path_plot=path_compare_distances + 'linear_distance'
                        )
                    df_to_file(
                        data=pd.DataFrame(linear),
                        path_file=path_compare_distances + 'linear_distance',
                        file_format_features=self.file_format_features
                    )

                if 'linear2' in self.compare_tasks:
                    linear2 = delta.functions.linear2(corpora)
                    if 'plot' in self.tasks:
                        heatmap_comparison(
                            data=pd.DataFrame(linear2),
                            title='linear2 distance',
                            path_plot=path_compare_distances + 'linear2_distance'
                        )
                    df_to_file(
                        data=pd.DataFrame(linear),
                        path_file=path_compare_distances + 'linear2_distance',
                        file_format_features=self.file_format_features
                    )

                if 'burrows2' in self.compare_tasks:
                    burrows2 = delta.functions.burrows2(corpora)
                    if 'plot' in self.tasks:
                        heatmap_comparison(
                            data=pd.DataFrame(burrows2),
                            title='linear2 distance',
                            path_plot=path_compare_distances + 'burrows2_distance'
                        )
                    df_to_file(
                        data=pd.DataFrame(burrows2),
                        path_file=path_compare_distances + 'burrows2_distance',
                        file_format_features=self.file_format_features
                    )

                if 'manhattan' in self.compare_tasks:
                    manhattan = delta.functions.manhattan(corpora)
                    if 'plot' in self.tasks:
                        heatmap_comparison(
                            data=pd.DataFrame(manhattan),
                            title='manhattan distance',
                            path_plot=path_compare_distances + 'manhattan_distance'
                        )
                    df_to_file(
                        data=pd.DataFrame(manhattan),
                        path_file=path_compare_distances + 'manhattan_distance',
                        file_format_features=self.file_format_features
                    )

                if 'euclidean' in self.compare_tasks:
                    euclidean = delta.functions.euclidean(corpora)
                    if 'plot' in self.tasks:
                        heatmap_comparison(
                            data=pd.DataFrame(euclidean),
                            title='euclidean distance',
                            path_plot=path_compare_distances + 'euclidean_distance'
                        )
                    df_to_file(
                        data=pd.DataFrame(euclidean),
                        path_file=path_compare_distances + 'euclidean_distance',
                        file_format_features=self.file_format_features
                    )

                if 'sqeuclidean' in self.compare_tasks:
                    sqeuclidean = delta.functions.sqeuclidean(corpora)
                    if 'plot' in self.tasks:
                        heatmap_comparison(
                            data=pd.DataFrame(sqeuclidean),
                            title='sqeuclidean distance',
                            path_plot=path_compare_distances + 'sqeuclidean_distance'
                        )
                    df_to_file(
                        data=pd.DataFrame(sqeuclidean),
                        path_file=path_compare_distances + 'sqeuclidean_distance',
                        file_format_features=self.file_format_features
                    )

                if 'cosine' in self.compare_tasks:
                    cosine = delta.functions.cosine(corpora)
                    if 'plot' in self.tasks:
                        heatmap_comparison(
                            data=pd.DataFrame(cosine),
                            title='cosine distance',
                            path_plot=path_compare_distances + 'cosine_distance'
                        )
                    df_to_file(
                        data=pd.DataFrame(cosine),
                        path_file=path_compare_distances + 'cosine_distance',
                        file_format_features=self.file_format_features
                    )

                if 'canberra' in self.compare_tasks:
                    canberra = delta.functions.canberra(corpora)
                    if 'plot' in self.tasks:
                        heatmap_comparison(
                            data=pd.DataFrame(canberra),
                            title='canberra distance',
                            path_plot=path_compare_distances + 'canberra_distance'
                        )
                    df_to_file(
                        data=pd.DataFrame(canberra),
                        path_file=path_compare_distances + 'canberra_distance',
                        file_format_features=self.file_format_features
                    )

                if 'braycurtis' in self.compare_tasks:
                    braycurtis = delta.functions.braycurtis(corpora)
                    if 'plot' in self.tasks:
                        heatmap_comparison(
                            data=pd.DataFrame(braycurtis),
                            title='braycurtis distance',
                            path_plot=path_compare_distances + 'braycurtis_distance'
                        )
                    df_to_file(
                        data=pd.DataFrame(braycurtis),
                        path_file=path_compare_distances + 'braycurtis_distance',
                        file_format_features=self.file_format_features
                    )
                if 'correlation' in self.compare_tasks:
                    correlation = delta.functions.correlation(corpora)
                    if 'plot' in self.tasks:
                        heatmap_comparison(
                            data=pd.DataFrame(correlation),
                            title='correlation distance',
                            path_plot=path_compare_distances + 'correlation_distance'
                        )
                    df_to_file(
                        data=pd.DataFrame(correlation),
                        path_file=path_compare_distances + 'correlation_distance',
                        file_format_features=self.file_format_features
                    )

                if 'chebyshev' in self.compare_tasks:
                    chebyshev = delta.functions.chebyshev(corpora)
                    if 'plot' in self.tasks:
                        heatmap_comparison(
                            data=pd.DataFrame(chebyshev),
                            title='chebyshev distance',
                            path_plot=path_compare_distances + 'chebyshev_distance'
                        )
                    df_to_file(
                        data=pd.DataFrame(chebyshev),
                        path_file=path_compare_distances + 'chebyshev_distance',
                        file_format_features=self.file_format_features
                    )

                if 'burrows' in self.compare_tasks:
                    burrows = delta.functions.burrows(corpora)
                    if 'plot' in self.tasks:
                        heatmap_comparison(
                            data=pd.DataFrame(burrows),
                            title='burrows distance',
                            path_plot=path_compare_distances + 'burrows_distance'
                        )
                    df_to_file(
                        data=pd.DataFrame(burrows),
                        path_file=path_compare_distances + 'burrows_distance',
                        file_format_features=self.file_format_features
                    )

                if 'quadratic' in self.compare_tasks:
                    quadratic = delta.functions.quadratic(corpora)
                    if 'plot' in self.tasks:
                        heatmap_comparison(
                            data=pd.DataFrame(quadratic),
                            title='quadratic distance',
                            path_plot=path_compare_distances + 'quadratic_distance'
                        )
                    df_to_file(
                        data=pd.DataFrame(quadratic),
                        path_file=path_compare_distances + 'quadratic_distance',
                        file_format_features=self.file_format_features
                    )

                if 'eder' in self.compare_tasks:
                    eder = delta.functions.eder(corpora)
                    if 'plot' in self.tasks:
                        heatmap_comparison(
                            data=pd.DataFrame(eder),
                            title='eder distance',
                            path_plot=path_compare_distances + 'eder_distance'
                        )
                    df_to_file(
                        data=pd.DataFrame(eder),
                        path_file=path_compare_distances + 'eder_distance',
                        file_format_features=self.file_format_features
                    )

                if 'cosine_delta' in self.compare_tasks:
                    cosine_delta = delta.functions.cosine_delta(corpora)
                    if 'plot' in self.tasks:
                        heatmap_comparison(
                            data=pd.DataFrame(cosine_delta),
                            title='cosine_delta distance',
                            path_plot=path_compare_distances + 'cosine_delta_distance'
                        )
                    df_to_file(
                        data=pd.DataFrame(cosine_delta),
                        path_file=path_compare_distances + 'cosine_delta_distance',
                        file_format_features=self.file_format_features
                    )



    def source_analytics(self):
        """Computation of comparison mode"""

        files = sorted(glob.glob(self.path_sources + os.sep + '*/*.json'))
        if not files:
            logging.warning(self.path_sources + " contains no files. Modus 'compare' not started.")

        features = {}
        features_col = {}

        for f in files:

            #logging.info('Open file with source: ' + f)
            corpus = f.replace(self.path_sources + os.sep, '').split(os.sep)[0]
            feature = f.replace(self.path_sources + os.sep, '').split(os.sep)[1].replace(corpus + '__', '').replace('.json', '')

            if feature not in features.keys():
                features[feature] = {}
                features_col[feature] = {}

            if corpus in self.corpora.keys():
                logging.info('Open file with source: ' + f)

                with (open(f, encoding='utf-8') as json_file):
                    content = json.load(json_file)
                    if content != {}:
                        features[feature][corpus] = content
                        if 'collection' in self.corpora[corpus]:

                            if self.corpora[corpus]['collection'] not in features_col[feature].keys():
                                features_col[feature][self.corpora[corpus]['collection']] = content
                                #features_col[feature][self.corpora[corpus]['collection']] = collections.Counter(content)
                            else:

                                features_col[feature][self.corpora[corpus]['collection']] = dict(list(features_col[feature][self.corpora[corpus]['collection']].items()) + list(content.items()))
                    else:
                        logging.info('Content of ' + f + ' is empty! It is not loaded.')

        self.source_analytics_corp_col(dict(features_col))
