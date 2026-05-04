from dopameter.language_collection import LanguageCollection, get_collection_sizes


class CorporaCollection(LanguageCollection):

    def __init__(self, lang, collection_name, features):
        self.lang = lang
        self.name = collection_name
        self.corpora = []

        self.documents = {}
        self.documents_pos = {}
        self.document_cnt_characteristics = {}

        from dopameter.language_collection.corpora_collection.corpora.corpus import init_scores
        self.features, self.macro_features, self.counts, self.resources, self.sizes\
            = init_scores(
                conf_features=features,
                lang=self.lang
            )

    def get_collection_counts(self):

        return get_collection_sizes(self.sizes)
