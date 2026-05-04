import collections


class Resources:

    def __init__ (self, conf_features):

        self.terminologies = {}
        self.ngrams = {}

        self.lexical_diversity = {}
        self.surface = {}

        self.syntax_dep = {}
        self.syntax_const = {}

        self.semantic_relations_wordnet = collections.Counter()
        self.emotion = {}

        self.semantic_relations_germanet = collections.Counter()

        if 'ngrams' in conf_features.keys():
            for n in conf_features['ngrams']:
                self.ngrams[n] = {}

        if 'lexical_diversity' in conf_features.keys():
            from dopameter.featurehub.lexical_diversity import init_lexical_diversity
            self.lexical_diversity = init_lexical_diversity()

        if 'surface' in conf_features.keys():
            from dopameter.featurehub.surface import init_surface
            self.surface = init_surface()

        if 'syntax_dependency_metrics' in conf_features.keys():
            from dopameter.featurehub.syntax.dependency import init_syntax_dependency_metrics
            self.syntax_dep = init_syntax_dependency_metrics()

        if 'syntax_constituency_metrics' in conf_features.keys():
            from dopameter.featurehub.syntax.constituency import init_syntax_constituency_metrics
            self.syntax_const = init_syntax_constituency_metrics()

        if 'emotion'  in conf_features.keys():
            from dopameter.featurehub.emotion import init_emotion
            self.emotion = init_emotion(conf_features['emotion'])

        for feat in (set(conf_features.keys())).intersection({'wordnet_synsets', 'wordnet_senses'}):
            self.terminologies[feat] = collections.Counter()

        for feat in (set(conf_features.keys())).intersection({'germanet_synsets', 'germanet_lexical_units', 'germanet_word_classes', 'germanet_semantic_relations'}):
            self.terminologies[feat] = collections.Counter()

    def update_resource_by_data(self, feature, data):

        if feature == 'lexical_diversity':
            from dopameter.featurehub.lexical_diversity import update_lexical_diversity
            self.lexical_diversity = update_lexical_diversity(
                lexical_diversity=self.lexical_diversity,
                data=data['lexical_diversity']
            )

        if feature == 'surface':
            from dopameter.featurehub.surface import update_surface
            self.surface = update_surface(
                surface=self.surface,
                data=data['surface']
            )

        if feature == 'syntax_dependency_metrics':
            from dopameter.featurehub.syntax.dependency import update_syntax_dependency_metrics
            self.syntax_dep = update_syntax_dependency_metrics(
                syntax_dep=self.syntax_dep,
                data=data['syntax_dep']
            )

        if feature == 'syntax_constituency_metrics':
            from dopameter.featurehub.syntax.constituency import update_syntax_constituency_metrics
            self.syntax_const = update_syntax_constituency_metrics(
                syntax_const=self.syntax_const,
                data=data['syntax_const']
            )

        if feature == 'wordnet_semantic_relations':
            self.semantic_relations_wordnet += data['doc']

        if feature == 'emotion':
            from dopameter.featurehub.emotion import update_emotion
            self.emotion = update_emotion(
                emotion=self.emotion,
                data=data['emotion']
            )

        if feature in ['wordnet_synsets', 'wordnet_senses']:
            #self.terminologies[feature].update(data['counts'])
            for key in data['counts']:
                if key not in self.terminologies[feature].keys():
                    self.terminologies[feature][key] = data['counts'][key]
                else:
                    self.terminologies[feature][key] = self.terminologies[feature][key] + data['counts'][key]

        if feature == 'ngrams':
            for n in data['ngrams'].keys():
                if n != 'counts':
                    for key in data['ngrams'][n]:
                        if key not in self.ngrams[n]:
                            self.ngrams[n][key] = data['ngrams'][n][key]
                        else:
                            self.ngrams[n][key] = self.ngrams[n][key] + data['ngrams'][n][key]

        if feature == 'germanet_semantic_relations':
            self.semantic_relations_germanet += data['doc']


    def update_resource_by_resources(self, feature, resources):

        if feature == 'lexical_diversity':
            from dopameter.featurehub.lexical_diversity import update_lexical_diversity
            self.lexical_diversity = update_lexical_diversity(
                lexical_diversity=self.lexical_diversity,
                data=resources.lexical_diversity
            )

        if feature == 'surface':
            from dopameter.featurehub.surface import update_surface
            self.surface = update_surface(
                surface=self.surface,
                data=resources.surface
            )

        if feature == 'syntax_dependency_metrics':
            from dopameter.featurehub.syntax.dependency import update_syntax_dependency_metrics
            self.syntax_dep = update_syntax_dependency_metrics(
                syntax_dep=self.syntax_dep,
                data=resources.syntax_dep
            )

        if feature == 'syntax_constituency_metrics':
            from dopameter.featurehub.syntax.constituency import update_syntax_constituency_metrics
            self.syntax_const = update_syntax_constituency_metrics(
                syntax_const=self.syntax_const,
                data=resources.syntax_const
            )

        if feature == 'wordnet_semantic_relations':
            self.semantic_relations_wordnet += resources.semantic_relations_wordnet

        if feature == 'emotion':
            from dopameter.featurehub.emotion import update_emotion
            self.emotion = update_emotion(
                emotion=self.emotion,
                data=resources.emotion
            )

        if feature in ['wordnet_synsets', 'wordnet_senses']:
            self.terminologies[feature].update(resources.terminologies[feature])

        if feature.startswith('dictionary_lookup'):
            self.terminologies[feature].update(resources.terminologies[feature])

        if feature == 'ngrams':
            for n in resources.ngrams:
                self.ngrams[n].update(resources.ngrams[n])

        if feature == 'germanet_semantic_relations':
            self.semantic_relations_germanet += resources.semantic_relations_germanet


    def clear(self):
        self.terminologies.clear()
        self.ngrams.clear()

        self.lexical_diversity.clear()
        self.surface.clear()

        self.syntax_dep.clear()
        self.syntax_const.clear()

        self.semantic_relations_wordnet.clear()
        self.emotion.clear()

        self.semantic_relations_germanet.clear()
