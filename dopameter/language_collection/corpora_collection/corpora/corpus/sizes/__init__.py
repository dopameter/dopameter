class Sizes:
    def __init__(self):

        self.types = set()
        self.lemmata = set()
        self.sentences = set()

        self.characters = 0
        self.sentences_cnt = 0
        self.tokens_cnt = 0

    def update_sizes_by_data(self, data):

        self.types.update(data['types'])
        self.lemmata.update(data['lemmata'])
        self.sentences.update(data['different_sentences'])

        self.characters += data['characters_cnt']
        self.sentences_cnt += data['sentences_cnt']
        self.tokens_cnt += data['tokens_cnt']

    def update_sizes_by_corpus_scores(self, sizes):

        self.types.update(sizes.types)
        self.lemmata.update(sizes.lemmata)
        self.sentences.update(sizes.sentences)

        self.characters += sizes.characters
        self.sentences_cnt += sizes.sentences_cnt
        self.tokens_cnt += sizes.tokens_cnt