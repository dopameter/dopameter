import logging

from dopameter.analytics.summarization import df_to_file, macro_df_to_file, sum_all_corpora_to_file, macro_features_to_file, features_to_file
from dopameter.analytics.vis_utils import matrix_to_boxplot
from dopameter.analytics.comparison import CompareAnalytics


class DoPaMeter:
    """DoPaMeter

    Parameters
    ----------

    config_corpora : dict
    config : dict


    Notes
    -----

    Here, all functionalities of DoPa Meter are convergence.
    For a deeper knowledge, read the code documentation under the path 'doc'.
    This class represents an automat, that combines all functionalities of DoPa Meter.
    It is derived into sections of 'features', 'counts' and 'corpus_characteristics',
    that is computed by a run through a nlp pipeline and a lot of modules of a hub o features.
    followed by 'plot', 'features_detail' and 'clusters'.
    """

    def __init__(
            self,
            config_corpora,
            config
    ):
        self.config_corpora = config_corpora
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

        for corpus_name in self.config_corpora['corpora'].keys():
            for c in ['<', '>', ':', '"', '/', "\\", '|', '?', '*']:
                if c in corpus_name:
                    self.config_corpora['corpora'][corpus_name]['internal_name'] = corpus_name.replace(c, '')

        if not set(tasks).intersection(valid_tasks):
            exit("Your given set of tasks is wrong. Allowed task definitions: " + ' '.join(valid_tasks) + ".")

        if 'compare' in tasks and 'features' not in tasks and 'counts' not in tasks and 'corpus_characteristics' not in tasks:
            if set(self.config['compare']).intersection({'bleu', 'nist', 'meteor'}):
                self.config['features'] = 'bleu'
                tasks += 'features'

        if set(tasks).intersection({'features', 'counts', 'corpus_characteristics'}):
            from dopameter.featurehub import process_feature_hub

            process_feature_hub(
                conf_corpora=self.config_corpora,
                config=self.config,
                tasks=tasks
            )

        config = {**self.config, **self.config_corpora}

        if 'compare' in tasks:
            from dopameter.analytics import comparison
            comparison(config=config)

        if 'features_detail' in tasks:
            from dopameter.analytics import detail_metrics
            detail_metrics(config=config)

        if 'features_detail' in tasks and 'plot' in tasks:
            from dopameter.analytics import visualize_detailed_features
            visualize_detailed_features(config=config)

        if 'cluster' in tasks:
            from dopameter.analytics.aggregation import aggregation
            aggregation(config=config)

        logging.info('===============================================================================================')
        logging.info('Running DoPa Meter done.')
        logging.info('===============================================================================================')

def features():
    return None