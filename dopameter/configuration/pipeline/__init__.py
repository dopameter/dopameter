import json
import os
import sys
import re
import logging
import spacy
import textacy
from spacy.tokens import Doc, Span
from dopameter.configuration.installation import ConfLanguages


class PreProcessingPipline:
    def __init__(
            self,
            config={}
    ):

        self.config_lang = json.load(open(os.path.join(os.sep.join(
            os.path.abspath(__file__).split(os.sep)[:-4]),
            'ext_res',
            'installation',
            'config_lang.json'))
        )

        self.spacy_languages = self.config_lang['spacy_languages']
        self.config = config

    def _get_tokens(self, span):
        """Get tokens as list

        Parameters
        ----------
        span : spaCy span

        Returns
        -------
        array of string
        """

        return [tok.text for tok in span if not re.match(r'\s+', tok.text)]

    def _get_pos(self, span):
        """Get lemma as list

        Parameters
        ----------
        span : spaCy span

        Returns
        -------
        array of string
        """

        return [tok.pos_ for tok in span if not re.match(r'\s+', tok.text)]

    def _get_n_tokens(self, span):
        """Get the number of tokens in span

        Parameters
        ----------
        span : spaCy span

        Returns
        -------
        int
        """
        return len(span._.tokens)

    def _get_n_sentences_in_doc(self, doc):
        """Get the number of sentences in span

        Parameters
        ----------
        doc : spaCy doc

        Returns
        -------
        int
        """
        return len([sent.text for sent in doc.sents])

    def _get_different_sentences_in_doc(self, doc):
        """Get the number of sentences in span

        Parameters
        ----------
        doc : spaCy doc

        Returns
        -------
        set
        """
        return set([sent.text for sent in doc.sents])

    def _get_n_tokens_punct(self, span):
        """Get the number of punctuation tokens in span

        Parameters
        ----------
        span : spaCy span

        Returns
        -------
        int
        """
        return sum(1 for token in span if token.is_punct)

    def _get_types_token(self, span):
        """Get the set of types in doc given by preprocessed tokens in an array

        Parameters
        ----------
        span : spaCy span

        Returns
        -------
        int
        """

        return set([
            str(ngram) for ngram in list(textacy.extract.ngrams(
                span,
                int(1),
                filter_stops=False,
                filter_punct=False,
                filter_nums=False))]
            )

    def _get_n_types_token(self, span):
        """Get the number of types in doc

        Parameters
        ----------
        span : spaCy span

        Returns
        -------
        int
        """
        return len(span._.vocab_types)

    def _get_lemmata(self, span):
        """Get the set of lemmata in doc given by preprocessed tokens in an array

        Parameters
        ----------
        span : spaCy span

        Returns
        -------
        int
        """

        return set([tok.lemma_ for tok in span])

    def _get_n_lemmata(self, span):
        """Get the number of lemmata in doc

        Parameters
        ----------
        span : spaCy span

        Returns
        -------
        int
        """
        return len(span._.lemmata)


    def _get_cnt_words(self, doc):
        """Get number of words of a document with filtered punctuation
            derived from https://github.com/mholtzscher/spacy_readability/blob/18ff66ae78299306733d987e509aaa0f775779b5/spacy_readability/__init__.py#L180

        Parameters
        ----------
        doc : spaCy Doc

        Returns
        -------
        int
            number of words of a document with filtered punctuation and
            words that start with apostrophe (aka contractions)
        """

        return len([word for word in doc if not word.is_punct and "'" not in word.text])

    def _get_cnt_syllables(self, doc, min_syllables: int = 1):
        """Get number of polysyllables by a given minimal syllable size.
            derived from https://github.com/mholtzscher/spacy_readability/blob/18ff66ae78299306733d987e509aaa0f775779b5/spacy_readability/__init__.py#L180

        Parameters
        ----------
        doc : spaCy Doc
        min_syllables : mininmal syllable size

        Returns
        -------
        int
            number of syllables of a document with filtered punctuation and
            words that start with apostrophe (aka contractions)
        """
        syllables_per_word = [0 if s is None else s for s in tuple(word._.syllables_count for word in (word for word in doc if not word.is_punct and "'" not in word.text))]
        return sum(c for c in syllables_per_word if c >= min_syllables)

    def _get_cnt_poly_syllables(self, doc, min_syllables: int = 3):
        return self._get_cnt_syllables(doc, min_syllables)


    def create_nlp(self, lang):
        """creates a spaCy nlp pipeline of a given language including extensions

        Parameters
        ----------
        lang

        Returns
        -------
        nlp: spaCy pipeline
        """

        if lang in self.spacy_languages.keys():
            logging.info("Create spaCy nlp module '" + self.spacy_languages[lang] + "' for language '" + self.config_lang['spacy_languages_def'][lang] + "' (" + lang + ")")
            nlp = spacy.load(ConfLanguages().spacy_languages[lang])

            if not Doc.has_extension('tokens'):
                Doc.set_extension('tokens', getter=self._get_tokens)
            if not Span.has_extension('tokens'):
                Span.set_extension('tokens', getter=self._get_tokens)

            if not Doc.has_extension('n_tokens'):
                Doc.set_extension('n_tokens', getter=self._get_n_tokens)
            if not Span.has_extension('n_tokens'):
                Span.set_extension('n_tokens', getter=self._get_n_tokens)

            if not Doc.has_extension('n_sentences_in_doc'):
                Doc.set_extension('n_sentences_in_doc', getter=self._get_n_sentences_in_doc)
            if not Span.has_extension('n_sentences_in_doc'):
                Span.set_extension('n_sentences_in_doc', getter=self._get_n_sentences_in_doc)

            if not Doc.has_extension('different_sentences_in_doc'):
                Doc.set_extension('different_sentences_in_doc', getter=self._get_different_sentences_in_doc)
            if not Span.has_extension('different_sentences_in_doc'):
                Span.set_extension('different_sentences_in_doc', getter=self._get_different_sentences_in_doc)

            if not Doc.has_extension('vocab_types'):
                Doc.set_extension('vocab_types', getter=self._get_types_token)
            if not Span.has_extension('vocab_types'):
                Span.set_extension('vocab_types', getter=self._get_types_token)

            if not Doc.has_extension('n_vocab_types'):
                Doc.set_extension('n_vocab_types', getter=self._get_n_types_token)
            if not Span.has_extension('n_vocab_types'):
                Span.set_extension('n_vocab_types', getter=self._get_n_types_token)

            if not Doc.has_extension('lemmata'):
                Doc.set_extension('lemmata', getter=self._get_lemmata)
            if not Span.has_extension('lemmata'):
                Span.set_extension('lemmata', getter=self._get_lemmata)

            if not Doc.has_extension('n_lemmata'):
                Doc.set_extension('n_lemmata', getter=self._get_n_lemmata)
            if not Span.has_extension('n_lemmata'):
                Span.set_extension('n_lemmata', getter=self._get_n_lemmata)

            if not Doc.has_extension('cnt_words'):
                Doc.set_extension('cnt_words', getter=self._get_cnt_words)
            if not Span.has_extension('cnt_words'):
                Span.set_extension('cnt_words', getter=self._get_cnt_words)

            if 'features' in self.config.keys():
                if 'surface' in self.config['features'].keys():
                    if not Doc.has_extension('cnt_syllables'):
                        Doc.set_extension('cnt_syllables', getter=self._get_cnt_syllables)
                    if not Span.has_extension('cnt_syllables'):
                        Span.set_extension('cnt_syllables', getter=self._get_cnt_syllables)

                    if not Doc.has_extension('cnt_poly_syllables'):
                        Doc.set_extension('cnt_poly_syllables', getter=self._get_cnt_poly_syllables)
                    if not Span.has_extension('cnt_poly_syllables'):
                        Span.set_extension('cnt_poly_syllables', getter=self._get_cnt_poly_syllables)

            if 'settings' in self.config.keys():
                if 'mode_ngrams' in self.config['settings'].keys() and self.config['settings']['mode_ngrams'] == 'pos':
                    if not Doc.has_extension('pos_tags'):
                        Doc.set_extension('pos_tags', getter=self._get_pos)
                    if not Span.has_extension('pos_tags'):
                        Span.set_extension('pos_tags', getter=self._get_pos)

            return nlp
        else:
            sys.exit('Your configured language ' + self.spacy_languages + ' is not supported. Check your config file.')
