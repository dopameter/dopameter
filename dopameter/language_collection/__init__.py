from dopameter.configuration.installation import ConfLanguages


class LanguageCollection:

    def __init__(self, lang, features):
        self.lang = lang
        self.name = ConfLanguages().lang_def[lang]
        self.collections = {}

        self.corpora = []

        self.documents = {}
        self.documents_pos = {}
        self.document_cnt_characteristics = {}

        from dopameter.language_collection.corpora_collection.corpora.corpus import init_scores
        self.features, self.macro_features, self.counts, self.resources, self.sizes = init_scores(conf_features=features, lang=self.lang)

    def update_scores(self, feature, resources):
        self.resources.update_resource_by_resources(feature=feature, resources=resources)

    def update_counts(self, feature, scores):
        if feature not in self.counts.keys() or self.counts[feature] == {}:
            self.counts[feature] = scores
        else:
            for key in scores:
                if key in self.counts[feature].keys():
                    self.counts[feature][key] += scores[key]
                else:
                    self.counts[feature][key] = scores[key]

    def clear(self):
        self.resources.clear()
        self.counts.clear()
        self.features.clear()