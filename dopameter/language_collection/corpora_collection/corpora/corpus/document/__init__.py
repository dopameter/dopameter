class BasicCharacteristics:
    def count_doc(self, doc):

        """

        Parameters
        ----------
        doc : spaCy document

        Returns
        -------
        dict including counts of 'characters_cnt', 'sentences_cnt', 'tokens_cnt', 'types', 'lemmata', 'different_sentences'
        """

        return {
            'characters_cnt':       len(doc.text),
            'sentences_cnt':        doc._.n_sentences_in_doc,
            'tokens_cnt':           doc._.n_tokens,
            'types':                doc._.vocab_types,
            'lemmata':              doc._.lemmata,
            'different_sentences':  doc._.different_sentences_in_doc
        }

