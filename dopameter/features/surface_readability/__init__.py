import collections
from collections import Counter
from math import sqrt
import numpy as np
import logging


class SurfaceFeaturizes:
    """Get metrics of surface patterns - not DE or EN languages

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
            'dale_chall'
        ]

    Attributes
    ----------
    * `avg_token_len_chars` : average token length measured by count of characters
    * `avg_sent_len_tokens` : average sentences length measured by count of tokens
    * `avg_sent_len_chars` : average sentences measured by count of characters

    * `avg_token_len_chars`: average token length measured by count of characters
    * `avg_sent_len_tokens`: average sentences length measured by count of tokens
    * `avg_sent_len_chars`: average sentences measured by count of characters

    * `flesch_reading_ease` : measured by Flesch reading ease formula (Rudolf Flesch 1948)
        :math:`[206.835 - 1.015 (count of tokens / count of sentences) - 84.6 (count of  syllables / count of words)]`
    * `dale_chall` : Dale-Chall score (Edgar Chall and Jeanne Chall, 1948)
        :math:`[0.1579 ((difficult words / words) * 100) + 0.0496 (words / sentences)]`

    * `flesch_kincaid_grade_level` : measured by Flesch-Kincaid grade level formula
        (Rudolf Flesch and Peter Kindaid, 1975, US Navy)
        :math:`[ 0.39 * (total words / total sentences) + 11.8 (total syllables / total words) - 15.59 ]`
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
    * `heylighen_formality` : F-Score (formality score as defined by Heylighen and Dewaele (1999)
        :math:`(noun freq + adjective freq + preposition freq + article freq - pronoun freq - verb freq - adverb freq - interjection freq + 100) / 2`

    Notes
    -----

    The surface level related metrics in this package is inspired by the spaCy extension 'Readability',
    what is implemented for an older spacy version (< 3)
    and English Language:
    https://spacy.io/universe/project/spacy_readability
    https://github.com/mholtzscher/spacy_readability/tree/master/spacy_readability


    Define in configuration .json file under features:

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
          "no_digit_tokens",
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
            features=['default']
    ):
        logging.info('\tInitialize surface / readability features.')

        self.features = features

        default_features = ['avg_token_len_chars', 'avg_sent_len_tokens', 'avg_sent_len_chars',
                            'flesch_kincaid_grade_level', 'smog', 'coleman_liau', 'ari', 'forcast', 'gunning_fog',
                            'heylighen_formality']

        if features == ['default']:
            self.features = default_features
        else:
            if set(self.features).intersection(default_features) == set():
                raise ValueError('Your surface / readability features ' + ' '.join(self.features) + ' are not defined! Allowed definitions: ', default_features)
            else:
                logging.info('\t\tDefined features: ' + features)
                self.features = features


    def flesch_kincaid_grade_level(self, cnt_sentences, cnt_syllables, cnt_words):
        """Get Flesch Kincaid grade level (Rudolf Flesch and Peter Kindaid, 1975, US Navy)
        Parameters
        ----------
        cnt_sentences : number of sentences (int)
        cnt_syllables : number of syllables (int)
        cnt_words : number of words (int)

        Returns
        ------
        int
            Flesch-Kincaid grade level score of a document
        """

        if cnt_sentences == 0 or cnt_words == 0 or cnt_syllables == 0:
            return 0
        else:
            return (
                ((11.8 * cnt_syllables) / cnt_words)
                +
                ((0.39 * cnt_words) / cnt_sentences)
                - 15.59
            )

    def smog(self, cnt_sentences, cnt_words, cnt_poly_syllables):
        """Get SMOG score for a document. If there are less than 30 sentences then
        it returns 0 because the formula significantly loses accuracy on small corpora.
        (G. Harry McLaughlin, 1969)

        Parameters
        ----------
        cnt_sentences : number of sentences (int)
        cnt_words : number of words (int)
        cnt_poly : number of poly syllables (int)

        Returns
        -------
        int
            SMOG score of a document
        """

        if cnt_sentences < 30 or cnt_words == 0:
            return 0
        else:
            return 1.0430 * sqrt(cnt_poly_syllables * (30 / cnt_sentences)) + 3.1291

    def coleman_liau(self, cnt_sentences, cnt_words, cnt_letter_tokens):
        """Coleman-Liau index of a document. (Meri Coleman and T.L. Liau, 1975)

        Parameters
        ----------
        cnt_sentences : number of sentences (int)
        cnt_words : number of words (int)
        cnt_letter_tokens : amount of words without punctuations and digits (int)

        Returns
        -------
        int
            Coleman-Liau index of a document
        """

        if cnt_words <= 0:
            return 0

        if cnt_letter_tokens <= 0:
            return 0
        else:
            letters_to_words = (cnt_letter_tokens / cnt_words) * 100
            sent_to_words = (cnt_sentences / cnt_words) * 100
            return 0.0588 * letters_to_words - 0.296 * sent_to_words - 15.8

    def ari(self, cnt_sentences, cnt_words, cnt_no_digit_tokens):
        """Get the Automated Readability Index of a document (Senter and Smith, 1967)

        Parameters
        ----------
        cnt_sentences : number of sentences (int)
        cnt_words : number of words (int)
        cnt_no_digit_tokens : amount of words without punctuations (int)

        Returns
        -------
        int
            Coleman-Liau index of a document
        """

        if cnt_words <= 0 or cnt_sentences <= 0:
            return 0
        else:
            letter_to_words = cnt_no_digit_tokens / cnt_words
            words_to_sents = cnt_words / cnt_sentences
            return 4.71 * letter_to_words + 0.5 * words_to_sents - 21.43

    def forcast(self, cnt_words, segments):
        """Get FORCAST readablilty score (US military, Caylor and Sticht, 1973)

        Parameters
        ----------
        cnt_words : number of words (int)
        segments : list of arrays of lists with 150 spaCy tokens

        Returns
        -------
        int
            FORCAST score of a document
        """

        if cnt_words < 150:
            return 0
        else:
            forcast_values = []
            for seg in segments:
                mono_syllabic = 0
                for tok in seg:
                    if tok._.syllables_count == 1:
                        mono_syllabic += 1
                forcast_values.append(20 - (mono_syllabic / 10))

            if forcast_values:
                return np.mean(forcast_values)
            else:
                return 0

    def gunning_fog(self, cnt_sentences, cnt_words, syllables_per_word):
        """Get Gunning fog index (Robert Guning, 1952)

        Parameters
        ----------
        cnt_sentences : number of sentences (int)
        cnt_words : number of words (int)
        syllables_per_word : array of syllables per word (array of int)

        Returns
        -------
        int
            Gunning fog index score of a document
        """

        complex_words = sum([tok >= 3 for tok in syllables_per_word])

        if cnt_sentences == 0 or cnt_words == 0:
            return 0
        else:
            gunning_fog = 0.4 * (
                    (cnt_words / cnt_sentences)
                    + 100 *
                    (complex_words / cnt_words)
            )

        return gunning_fog

    def get_heylighen_formality_score(self, cnt_pos):
        """Roughly calculate F-score (formality score) based on Universal Dependencies v2 POS tag set

        F-Score defined by:
        Heylighen, Francis; Dewaele, Jean-Marc. 1999. Formality of Language: definition, measurement and behavioral determinants.
        Technical Report Internal Report, Center ”Leo Apostel”, Free University of Brussels, Brussels.

        Parameters
        ----------
        cnt_pos : Counter of frequency of part-of-speech elements

        Returns
        -------
        int
            score between 0 and 100
            0 if no F-Score could be calculated (F-Score never reaches this limit)
        """

        total_words = sum(v for v in cnt_pos.values())
        return (
                       (
                               (
                                       sum(v for k, v in cnt_pos.items() if k in {
                                           'NOUN', 'PROPN', 'ADJ', 'DET', 'ADP', 'PRON'
                                       })
                                       -
                                       sum(v for k, v in cnt_pos.items() if k in {
                                           'VERB', 'AUX', 'ADV', 'INTJ'
                                       })
                               )
                               /
                               total_words * 100
                       ) + 100
               ) / 2 if total_words else 0

    def feat_doc(self, doc):
        """Featurize a document with surface level metrics

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

        if 'avg_sent_len_tokens' in self.features:
            if sent_len_tokens:
                data['features']['avg_sent_len_tokens'] = np.mean(sent_len_tokens)
            else:
                data['features']['avg_sent_len_tokens'] = 0

        if 'avg_sent_len_chars' in self.features:
            if sent_len_chars:
                data['features']['avg_sent_len_chars'] = np.mean(sent_len_chars)
            else:
                data['features']['avg_sent_len_chars'] = 0

        cnt_sentences = len(list(doc.sents))
        cnt_letter_tokens = sum([len(token) for token in doc if not token.is_punct and not token.is_digit])
        cnt_no_digit_tokens = sum([len(token) for token in doc if not token.is_punct])

        syllables_per_word = [0 if s is None else s for s in [token._.syllables_count for token in doc]]

        toks_min_three_syllables = [token.text for token in doc if token._.syllables is not None and len(token._.syllables) >= 3]
        toks_larger_six_letters = [tok.text for tok in doc if len(tok.text) > 6]
        toks_one_syllable = [token.text for token in doc if token._.syllables is not None and len(token._.syllables) == 1]#[tok == 1 for tok in syllables_per_word]

        if 'toks_min_three_syllables' in self.features:
            data['features']['toks_min_three_syllables'] = len(toks_min_three_syllables) / len(doc)

        if 'toks_larger_six_letters' in self.features:
            data['features']['toks_larger_six_letters'] = len(toks_larger_six_letters) / len(doc)

        if 'toks_one_syllable' in self.features:
            data['features']['toks_one_syllable'] = len(toks_one_syllable) / len(doc)

        if 'flesch_kincaid_grade_level' in self.features:
            data['features']['flesch_kincaid_grade_level'] = self.flesch_kincaid_grade_level(cnt_sentences, doc._.cnt_syllables, doc._.cnt_words)

        if 'smog' in self.features:
            data['features']['smog'] = self.smog(cnt_sentences, doc._.cnt_words, doc._.cnt_poly_syllables)

        if 'coleman_liau' in self.features:
            data['features']['coleman_liau'] = self.coleman_liau(cnt_sentences, doc._.cnt_words, cnt_letter_tokens)

        if 'ari' in self.features:
            data['features']['ari'] = self.ari(cnt_sentences, doc._.cnt_words, cnt_no_digit_tokens)

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
            data['features']['forcast'] = self.forcast(doc._.cnt_words, segments)

        if 'gunning_fog' in self.features:
            data['features']['gunning_fog'] = self.gunning_fog(cnt_sentences, doc._.cnt_words, syllables_per_word)

        cnt_pos = Counter([token.pos_ for token in doc if token.pos_ in {
            'NOUN', 'PROPN', 'ADJ', 'DET', 'ADP', 'PRON', 'VERB', 'AUX', 'ADV', 'INTJ'
        }])
        if 'heylighen_formality' in self.features:
            data['features']['heylighen_formality'] = self.get_heylighen_formality_score(cnt_pos)

        if 'flesch_reading_ease' in self.features:
            self.features.remove('flesch_reading_ease')

        data['surface']['toks_min_three_syllables'] = collections.Counter(toks_min_three_syllables)
        data['surface']['toks_larger_six_letters'] = collections.Counter(toks_larger_six_letters)
        data['surface']['toks_one_syllable'] = collections.Counter(toks_one_syllable)

        data['surface']['token_len_chars'] = token_len_chars
        data['surface']['sent_len_tokens'] = sent_len_tokens
        data['surface']['sent_len_chars'] = sent_len_chars

        data['surface']['cnt_syllables'] = doc._.cnt_syllables
        data['surface']['cnt_words'] = doc._.cnt_words
        data['surface']['cnt_poly_syllables'] = doc._.cnt_poly_syllables
        data['surface']['cnt_letter_tokens'] = cnt_letter_tokens
        data['surface']['cnt_no_digit_tokens'] = cnt_no_digit_tokens

        data['surface']['syllables_per_word'] = syllables_per_word
        data['surface']['cnt_pos'] = cnt_pos

        data['surface']['segments'] = segments

        data['counts']['toks_min_three_syllables'] = len(toks_min_three_syllables)
        data['counts']['toks_larger_six_letters'] = len(toks_larger_six_letters)
        data['counts']['toks_one_syllable'] = len(toks_one_syllable)

        data['counts']['syllables'] = sum(data['surface']['syllables'].values())
        data['counts']['letter_tokens'] = sum(data['surface']['letter_tokens'].values())
        data['counts']['no_digit_tokens'] = sum(data['surface']['no_digit_tokens'].values())

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

        new_list = []
        for el in corpus.surface['segments']:
            new_list.extend(el)

        corpus_segments = []
        temp = []
        for i, tok in enumerate(new_list):
            if tok.text != '\n':
                temp.append(tok)
            if len(temp) == 150:
                corpus_segments.append(temp)
                temp = []
        if temp:
            corpus_segments.append(temp)

        if 'avg_token_len_chars' in self.features:
            if corpus.surface['token_len_chars']:
                data['features']['avg_token_len_chars'] = np.mean(corpus.surface['token_len_chars)'])
            else:
                data['features']['avg_token_len_chars'] = 0

        if 'avg_sent_len_tokens' in self.features:
            if corpus.surface['sent_len_tokens']:
                data['features']['avg_sent_len_tokens'] = np.mean(corpus.surface['sent_len_tokens)'])
            else:
                data['features']['avg_sent_len_tokens'] = 0

        if 'avg_sent_len_chars' in self.features:
            if corpus.surface['sent_len_chars']:
                data['features']['avg_sent_len_chars'] = np.mean(corpus.surface['sent_len_chars)'])
            else:
                data['features']['avg_sent_len_chars'] = 0

        if 'flesch_kincaid_grade_level' in self.features:
            data['features']['flesch_kincaid_grade_level'] = self.flesch_kincaid_grade_level(
                corpus.surface['cnt_sentences'],
                corpus.surface['cnt_syllables'],
                corpus.surface['cnt_words']
            )

        if 'smog' in self.features:
            data['features']['smog'] = self.smog(
                corpus.surface['cnt_sentences'],
                corpus.surface['cnt_words'],
                corpus.surface['cnt_poly_syllables']
            )

        if 'coleman_liau' in self.features:
            data['features']['coleman_liau'] = self.coleman_liau(
                corpus.surface['cnt_sentences'],
                corpus.surface['cnt_words'],
                corpus.surface['cnt_letter_tokens']
            )

        if 'ari' in self.features:
            data['features']['ari'] = self.ari(
                corpus.surface['cnt_sentences'],
                corpus.surface['cnt_words'],
                corpus.surface['cnt_no_digit_tokens']
            )

        if 'forcast' in self.features:
            data['features']['forcast'] = self.forcast(
                corpus.surface['cnt_words'],
                corpus.surface['corpus_segments']
            )

        if 'gunning_fog' in self.features:
            data['features']['gunning_fog'] = self.gunning_fog(
                corpus.surface['cnt_sentences'],
                corpus.surface['cnt_words'],
                corpus.surface['syllables_per_word']
            )

        if 'heylighen_formality' in self.features:
            data['features']['heylighen_formality'] = self.get_heylighen_formality_score( corpus.surface['cnt_pos'])

        return data
