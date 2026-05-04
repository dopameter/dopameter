import collections
import json
import os

import numpy as np
import logging
from collections import Counter
from ...surface import SurfaceFeaturizes


class SurfaceFeaturizesDE(SurfaceFeaturizes):
    """Get metrics of surface patterns - German language

    Parameters
    ----------
    features : list, [
            'avg_token_len_chars',
            'avg_sent_len_tokens',
            'avg_sent_len_chars',
            'flesch_reading_ease',
            'flesch_kincaid_grade_level',
            'smog',
            'coleman_liau',
            'ari',
            'forcast',
            'gunning_fog',
            'heylighen_formality',
            'wiener_sachtextformel_1',
            'wiener_sachtextformel_2',
            'wiener_sachtextformel_3',
            'wiener_sachtextformel_4'
        ]

    Attributes
    ----------
    * `avg_token_len_chars` : average token length measured by count of characters
    * `avg_sent_len_tokens` : average sentences length measured by count of tokens
    * `avg_sent_len_chars` : average sentences measured by count of characters
    * `avg_token_len_chars`: average token length measured by count of characters
    * `avg_sent_len_tokens`: average sentences length measured by count of tokens
    * `avg_sent_len_chars`: average sentences measured by count of characters
    * `flesch_reading_ease` : measured by Flesch reading ease formula,
            only computed for German language and English language
            English (Rudolf Flesch 1948)
                :math:`[ 206.835 - 1.015 (count of tokens / count of sentences) - 84.6 (count of  syllables / count of words) ]`
            German (Toni Armstad 1978)
                :math:`[ 180 - (count of tokens / count of  sentences) - 58.5 (count of  syllables / count of  words) ]`
    * `flesch_kincaid_grade_level` : measured by Flesch-Kincaid grade level formula
            (Rudolf Flesch and Peter Kindaid, 1975, US Navy)
            :math:`[ 0.39 * (total words / total sentences) + 11.8 (total syllables / total words) - 15.59 ]`
        dale_chall : Dale-Chall score (Edgar Chall and Jeanne Chall, 1948), only English language
    * `smog` : SMOG Grading index (G. Harry McLaughlin, 1969) (only with document level)
            :math:`[ 1,0430 sqrt{count of polysyllables * (30 / count of sentences)} + 3.1291 ]`
    * `coleman_liau` : Coleman-Liau index (Meri Coleman and T.L. Liau, 1975)
            :math:`[ 0.0588 * (average number of letters per 100 words)
            - 0.296 * (average number of sentences per 100 words) - 15.8 ]`
    * `ari` : Automated readability index (R. J. Senter and E. A. Smith, 1967)
            :math:`[ 4.71 (characters / tokens) + 0.5 (tokens / sentences) - 21.43 ]`
    * `forcast` : FORCAST formula (US military, 1973) (only with document level)
            :math:`[ 20 - ((number of single-syllable words in a 150 word sample) / 10) ]`
    * `gunning_fog` : Gunning fog index (Robert Guning, 1952)
            :math:`[ 0.4 * ((words / sentences) + 100 * (complex words / word)) ]`
    * `wiener_sachtextformel_1` : First Wiener Sachtextformel (Richard Bamberger and Erich Vanecek, 1984)
            :math:`[ (0.1935 * words with 3 or more syllables) + (0.16772 * average sentence length) +
                (0.1297 * words with more than 6 letters) - (0.0327 * es) - 0.875 ]`
    * `wiener_sachtextformel_2` : Second Wiener Sachtextformel (Richard Bamberger and Erich Vanecek, 1984)
            :math:`[ (0.2007 * words with 3 or more syllables) + (0.1682 * average sentence length) +
                (0.1373 * words with more than 6 letters) - 2.779 ]`
    * `wiener_sachtextformel_3` : Third Wiener Sachtextformel (Richard Bamberger and Erich Vanecek, 1984)
            :math:`[ (0.2963 * words with 3 or more syllables) + (0.1905 * average sentence length) - 1.1144 ]`
    * `wiener_sachtextformel_4` : Fourth Wiener Sachtextformel (Richard Bamberger and Erich Vanecek, 1984)
            :math:`[ 0.2744 * (words with 3 or more syllables + 0.2656) * (average sentence length - 1.693) ]`
    * `heylighen_formality` : F-Score (formality score as defined by Heylighen and Dewaele (1999)
            :math:`(noun freq + adjective freq + preposition freq + article freq - pronoun freq - verb freq - adverb freq - interjection freq + 100) / 2`

    Notes
    -----

    define in configuration .json file under features:

    * default:
      "tasks": ["features", "counts"],
      "features": {
        "surface": "default"
      }
    * or in detail:
      "tasks": ["features", "counts"],
      "features": {
        "surface": [
          "toks_min_three_syllables",
          "toks_larger_six_letters",
          "toks_one_syllable",
          "syllables",
          "letter_tokens",
          "no_punct_tokens",
          "avg_token_len_chars",
          "avg_sent_len_tokens",
          "avg_sent_len_chars",
          "flesch_kincaid_grade_level",
          "smog",
          "coleman_liau",
          "ari",
          "forcast",
          "gunning_fog",
          "heylighen_formality"
          ]
        }
    """

    def __init__(
            self,
            features='default'
    ):

        default_features = [
            'avg_token_len_chars',
            'avg_sent_len_tokens',
            'avg_sent_len_chars',
            'flesch_reading_ease',
            'flesch_kincaid_grade_level',
            'smog',
            'coleman_liau',
            'ari',
            'forcast',
            'gunning_fog',
            'heylighen_formality',
            'wiener_sachtextformel_1',
            'wiener_sachtextformel_2',
            'wiener_sachtextformel_3',
            'wiener_sachtextformel_4'
        ]

        if features == 'default':
            self.features = default_features
        else:
            if set(self.features).intersection(default_features) == set():
                raise ValueError('Your surface / readability features ' + ' '.join(self.features) + ' are not defined! Allowed definitions: ', default_features)
            else:
                logging.info('\t\tDefined features: ' + features)
                self.features = features
        self.syllables = {}

    def flesch_reading_ease(self, cnt_sentences, cnt_words, cnt_syllables):
        """Get German Flesch-Kincaid Reading Ease score
            (Rudolf Flesch and Peter Kindaid, 1975, US Navy and Toni Armstad 1978)

        Parameters
        ----------
        cnt_sentences : number of sentences (int)
        cnt_words : number of words (int)
        cnt_syllables : number of syllables (int)

        Returns
        ------
        int
            German Flesch-Kincaid Reading Ease score of a document
        """

        if cnt_sentences == 0 or cnt_words == 0 or cnt_syllables == 0:
            return 0
        words_per_sent = cnt_words / cnt_syllables
        syllables_per_word = cnt_syllables / cnt_words

        return 180 - words_per_sent - (58.5 * syllables_per_word)

    def wiener_sachtext(self, len_doc, sent_lenghts, cnts_toks_min_three_syllables, cnts_toks_larger_six_letters, cnts_toks_one_syllable):
        """Get values of the four Wiener Sachtext formulas (Richard Bamberger and Erich Vanecek, 1984)

        Parameters
        ----------
        len_doc : length of a document (int)
        sent_lenghts : array of sentences lenghts of documents (array of int)
        cnts_toks_min_three_syllables : token with minimum of 3 syllables
        cnts_toks_larger_six_letters : array of tokens larger than six characters (array of string)
        cnts_toks_one_syllable : tokens with exact 1 syllable

        Returns
        -------
        tuple (int, int, int, int)
            tuple with values of all four Wiener Sachtext formulas
            (first Wiener Sachtext formula, second Wiener Sachtext formula,
            third Wiener Sachtext formula, fourth Wiener Sachtext formula)
        """

        if len_doc > 0:
            ms = (cnts_toks_min_three_syllables / len_doc) * 100
            sl = np.mean(sent_lenghts)
            iw = (cnts_toks_larger_six_letters / len_doc) * 100
            es = (cnts_toks_one_syllable / len_doc) * 100
        else:
            return 0

        wstf_1 = (0.1935 * ms) + (0.1672 * sl) + (0.1297 * iw) - (0.0327 * es) - 0.875
        wstf_2 = (0.2007 * ms) + (0.1682 * sl) + (0.1373 * iw) - 2.779
        wstf_3 = (0.2963 * ms) + (0.1905 * sl) - 1.1144
        wstf_4 = (0.2744 * ms) + (0.2656 * sl) - 1.693

        return wstf_1, wstf_2, wstf_3, wstf_4

    def feat_doc(self, doc):
        """Compute metrics of surface patterns for a document

        Parameters
        ----------
        doc : spaCy Doc

        Returns
        -------
        dict
            dictionary with 'features' as key including the metrics and 'surface' with interim results and 'counts' including amount of countable metrics
        """

        data = {'features': {}, 'counts':{}, 'surface': {}}

        token_len_chars = [len(token.text) for token in doc]
        sent_len_tokens = [len(sent) for sent in doc.sents]
        sent_len_chars = [len(sent.text) for sent in doc.sents]

        if 'avg_token_len_chars' in self.features:
            if token_len_chars:
                data['features']['avg_token_len_chars'] = np.mean(token_len_chars)
            else:
                data['features']['avg_token_len_chars'] = 0

        data['surface']['token_len_chars'] = token_len_chars


        if 'avg_sent_len_tokens' in self.features:
            if sent_len_tokens:
                data['features']['avg_sent_len_tokens'] = np.mean(sent_len_tokens)
            else:
                data['features']['avg_sent_len_tokens'] = 0

        data['surface']['sent_len_tokens'] = sent_len_tokens

        if 'avg_sent_len_chars' in self.features:
            if sent_len_chars:
                data['features']['avg_sent_len_chars'] = np.mean(sent_len_chars)
            else:
                data['features']['avg_sent_len_chars'] = 0

        data['surface']['sent_len_chars'] = sent_len_chars

        cnt_sentences = len(list(doc.sents))
        cnt_no_punct_tokens = sum([len(token) for token in doc if not token.is_punct])
        #cnt_letter_tokens = sum([len(token) for token in doc if not token.is_punct and not token.is_digit])
        cnt_letter_tokens = sum([len(token) for token in doc if not token.is_punct and not token.is_digit])

        syllables_per_word = [0 if s is None else s for s in [token._.syllables_count for token in doc]]

        # Inputs @ Wiener Sachtextformulas
        cnt_toks_min_three_syllables = len([token.text for token in doc if token._.syllables is not None and len(token._.syllables) >= 3])
        cnt_toks_larger_six_letters = len([tok.text for tok in doc if len(tok.text) > 6])
        cnt_toks_one_syllable = len([token.text for token in doc if token._.syllables is not None and len(token._.syllables) == 1])#[tok == 1 for tok in syllables_per_word]

        if 'toks_min_three_syllables' in self.features:
            data['features']['toks_min_three_syllables'] = cnt_toks_min_three_syllables / len(doc)

        if 'toks_larger_six_letters' in self.features:
            data['features']['toks_larger_six_letters'] = cnt_toks_larger_six_letters / len(doc)

        if 'toks_one_syllable' in self.features:
            data['features']['toks_one_syllable'] = cnt_toks_one_syllable / len(doc)

        if 'flesch_kincaid_grade_level' in self.features:
            data['features']['flesch_kincaid_grade_level'] = self.flesch_kincaid_grade_level(cnt_sentences, doc._.cnt_syllables, doc._.cnt_words)

        if 'smog' in self.features:
            data['features']['smog'] = self.smog(cnt_sentences, doc._.cnt_words, doc._.cnt_poly_syllables)

        if 'coleman_liau' in self.features:
            data['features']['coleman_liau'] = self.coleman_liau(cnt_sentences, doc._.cnt_words, cnt_letter_tokens)

        if 'ari' in self.features:
            data['features']['ari'] = self.ari(cnt_sentences, doc._.cnt_words, cnt_no_punct_tokens)

        temp = []
        segments = []

        for i, tok in enumerate(doc):
            if tok.text != '\n':
                temp.append(tok)
            if len(temp) == 150:
                segments.append(temp)
                temp = []
        if temp:
            segments.append(temp)

        if 'forcast' in self.features:
            data['features']['forcast'], data['surface']['forcast_values'] = self.forcast(doc._.cnt_words, segments)

        if 'gunning_fog' in self.features:
            data['features']['gunning_fog'] = self.gunning_fog(cnt_sentences, doc._.cnt_words, syllables_per_word)

        wstf_1, wstf_2, wstf_3, wstf_4 = self.wiener_sachtext(
            doc._.n_tokens,
            sent_len_tokens,
            cnt_toks_min_three_syllables,
            cnt_toks_larger_six_letters,
            cnt_toks_one_syllable
        )

        if 'wiener_sachtextformel_1' in self.features:
            data['features']['wiener_sachtextformel_1'] = wstf_1

        if 'wiener_sachtextformel_2' in self.features:
            data['features']['wiener_sachtextformel_2'] = wstf_2

        if 'wiener_sachtextformel_3' in self.features:
            data['features']['wiener_sachtextformel_3'] = wstf_3

        if 'wiener_sachtextformel_4' in self.features:
            data['features']['wiener_sachtextformel_4'] = wstf_4

        data['features']['flesch_reading_ease'] = self.flesch_reading_ease(cnt_sentences, doc._.cnt_words, doc._.cnt_syllables)

        cnt_pos = Counter([token.pos_ for token in doc if token.pos_ in {
                    'NOUN', 'PROPN', 'ADJ', 'DET', 'ADP', 'PRON', 'VERB', 'AUX', 'ADV', 'INTJ'
                }])
        if 'heylighen_formality' in self.features:
            data['features']['heylighen_formality'] = self.get_heylighen_formality_score(cnt_pos)

        data['surface']['cnt_syllables'] = doc._.cnt_syllables # für Flesch-Metriken
        data['surface']['cnt_words'] = doc._.cnt_words # number of words of a document with filtered punctuation
        data['surface']['cnt_poly_syllables'] = doc._.cnt_poly_syllables
        data['surface']['cnt_letter_tokens'] = cnt_letter_tokens
        data['counts']['letter_tokens'] = cnt_letter_tokens  # sum(data['surface']['letter_tokens'].values())

        data['surface']['syllables_per_word'] = syllables_per_word
        data['surface']['cnt_pos'] = cnt_pos

        data['surface']['syllables'] = collections.Counter([i for g in [word._.syllables for word in (word for word in doc if not word.is_punct and "'" not in word.text and word._.syllables != None)] for i in g])
        data['counts']['syllables'] = sum(data['surface']['syllables'].values())

        data['surface']['cnt_toks_min_three_syllables'] = cnt_toks_min_three_syllables
        data['counts']['toks_min_three_syllables'] = cnt_toks_min_three_syllables

        data['surface']['cnt_toks_larger_six_letters'] = cnt_toks_larger_six_letters
        data['counts']['toks_larger_six_letters'] = cnt_toks_larger_six_letters

        data['surface']['cnt_toks_one_syllable'] = cnt_toks_one_syllable
        data['counts']['toks_one_syllable'] = cnt_toks_one_syllable

        data['surface']['cnt_no_punct_tokens'] = cnt_no_punct_tokens
        data['counts']['no_punct_tokens'] = cnt_no_punct_tokens #sum(data['surface']['no_punct_tokens'].values())

        return data


    def feat_corpus(self, corpus):

        """Featurize a corpus with metrics of surface patterns

        Parameters
        ----------
        corpus

        Returns
        -------
        dict
            dictionary with 'features' as key including the corpus metrics
        """

        data = {'features': {}}

        if 'avg_token_len_chars' in self.features:
            if corpus.resources.surface['token_len_chars']:
                data['features']['avg_token_len_chars'] = np.mean(corpus.resources.surface['token_len_chars'])
            else:
                data['features']['avg_token_len_chars'] = 0

        if 'avg_sent_len_tokens' in self.features:
            if corpus.resources.surface['sent_len_tokens']:
                data['features']['avg_sent_len_tokens'] = np.mean(corpus.resources.surface['sent_len_tokens'])
            else:
                data['features']['avg_sent_len_tokens'] = 0

        if 'avg_sent_len_chars' in self.features:
            if corpus.resources.surface['sent_len_chars']:
                data['features']['avg_sent_len_chars'] = np.mean(corpus.resources.surface['sent_len_chars'])
            else:
                data['features']['avg_sent_len_chars'] = 0

        if 'toks_min_three_syllables' in self.features:
            if corpus.resources.surface['toks_min_three_syllables']:
                data['features']['toks_min_three_syllables'] = sum(corpus.resources.surface['toks_min_three_syllables'].values()) / corpus.resources.surface['cnt_words']
            else:
                data['features']['toks_min_three_syllables'] = 0

        if 'toks_larger_six_letters' in self.features:
            if corpus.resources.surface['toks_larger_six_letters']:
                data['features']['toks_larger_six_letters'] = sum(corpus.resources.surface['toks_larger_six_letters'].values()) / corpus.resources.surface['cnt_words']
            else:
                data['features']['toks_larger_six_letters'] = 0

        if 'toks_one_syllable' in self.features:
            if corpus.resources.surface['toks_one_syllable']:
                data['features']['toks_one_syllable'] = sum(corpus.resources.surface['toks_one_syllable'].values()) / corpus.resources.surface['cnt_words']
            else:
                data['features']['toks_one_syllable'] = 0

        if 'flesch_kincaid_grade_level' in self.features:
            data['features']['flesch_kincaid_grade_level'] = self.flesch_kincaid_grade_level(
                corpus.sizes.sentences_cnt,
                corpus.resources.surface['cnt_syllables'],
                corpus.resources.surface['cnt_words']
            )

        if 'smog' in self.features:
            data['features']['smog'] = self.smog(
                corpus.sizes.sentences_cnt,
                corpus.resources.surface['cnt_words'],
                corpus.resources.surface['cnt_poly_syllables']
            )

        if 'coleman_liau' in self.features:

            data['features']['coleman_liau'] = self.coleman_liau(
                corpus.sizes.sentences_cnt,
                corpus.resources.surface['cnt_words'],
                corpus.resources.surface['cnt_letter_tokens']
            )

        if 'ari' in self.features:
            data['features']['ari'] = self.ari(
                corpus.sizes.sentences_cnt,
                corpus.resources.surface['cnt_words'],
                corpus.resources.surface['cnt_no_punct_tokens']
            )

        if 'forcast' in self.features:
            data['features']['forcast'] = np.mean(corpus.resources.surface['forcast_values'])

        if 'gunning_fog' in self.features:
            data['features']['gunning_fog'] = self.gunning_fog(
                corpus.sizes.sentences_cnt,
                corpus.resources.surface['cnt_words'],
                corpus.resources.surface['syllables_per_word']
            )

        wstf_1, wstf_2, wstf_3, wstf_4 = self.wiener_sachtext(

            corpus.sizes.tokens_cnt,
            corpus.resources.surface['sent_len_tokens'],
            corpus.resources.surface['cnt_toks_min_three_syllables'],
            corpus.resources.surface['cnt_toks_larger_six_letters'],
            corpus.resources.surface['cnt_toks_one_syllable']
        )

        if 'wiener_sachtextformel_1' in self.features:
            data['features']['wiener_sachtextformel_1'] = wstf_1

        if 'wiener_sachtextformel_2' in self.features:
            data['features']['wiener_sachtextformel_2'] = wstf_2

        if 'wiener_sachtextformel_3' in self.features:
            data['features']['wiener_sachtextformel_3'] = wstf_3

        if 'wiener_sachtextformel_4' in self.features:
            data['features']['wiener_sachtextformel_4'] = wstf_4

        if 'flesch_reading_ease' in self.features:
            data['features']['flesch_reading_ease'] = self.flesch_reading_ease(
                corpus.sizes.sentences_cnt,
                corpus.resources.surface['cnt_words'],
                corpus.resources.surface['cnt_syllables']
            )

        if 'heylighen_formality' in self.features:
            data['features']['heylighen_formality'] = self.get_heylighen_formality_score(corpus.resources.surface['cnt_pos'])

        return data
