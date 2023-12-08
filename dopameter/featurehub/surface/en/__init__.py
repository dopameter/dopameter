import collections
import os
from collections import Counter
import numpy as np
import logging

from ...surface import SurfaceFeaturizes


class SurfaceFeaturizesEN(SurfaceFeaturizes):
    """Get metrics of surface patterns - English language

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
    * `dale_chall` : Dale-Chall score (Edgar Chall and Jeanne Chall, 1948), only English language
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
          "heylighen_formality",
          "dale_chall"
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
            'dale_chall',
            'smog',
            'coleman_liau',
            'ari',
            'forcast',
            'gunning_fog',
            'heylighen_formality'
            ]

        if features == 'default':
            self.features = default_features
        else:
            if set(self.features).intersection(default_features) == set():
                raise ValueError('Your surface / readability features ' + ' '.join(self.features) +
                                 ' are not defined! Allowed definitions: ', default_features)
            else:
                logging.info('\t\tDefined features: ' + features)
                self.features = features

        chall_file = open(os.path.join(os.sep.join(
                os.path.abspath(__file__).split(os.sep)[:-5]),
                'ext_res',
                'surface',
                'Dale_Chall_Words.dic')
            )

        self.DALE_CHALL_WORDS = set(line.strip() for line in chall_file)
        chall_file.close()

    def flesch_reading_ease(self, cnt_sentences, cnt_words, cnt_syllables):
        """Get English Flesch-Kincaid Reading Ease score (Rudolf Flesch and Peter Kindaid, 1975)

        Parameters
        ----------
        cnt_sentences : number of sentences (int)
        cnt_words : number of words (int)
        cnt_syllables : number of syllables (int)

        Returns
        ------
        int
            English Flesch-Kincaid Reading Ease score of a document
        """

        if cnt_sentences == 0 or cnt_words == 0 or cnt_syllables == 0:
            return 0
        else:
            words_per_sent = cnt_words / cnt_sentences
            syllables_per_word = cnt_syllables / cnt_words

        return 206.835 - (1.015 * words_per_sent) - (84.6 * syllables_per_word)

    def dale_chall(self, cnt_sentences, cnt_words, cnt_diff_words):
        """Get Dale-Chall score of a document (Edgar Chall and Jeanne Chall, 1948)

        Parameters
        ----------
        cnt_sentences : number of sentences (int)
        cnt_words : number of words (int)
        cnt_diff_words : number of different words (int)

        Returns
        -------
        int
            Get Dale-Chall score of a document
        """

        if cnt_sentences == 0 or cnt_words == 0:
            return 0

        if cnt_words != 0 and cnt_sentences != 0:
            percent_difficult_words = 100 * cnt_diff_words / cnt_words
            average_sentence_length = cnt_words / cnt_sentences
            grade = 0.1579 * percent_difficult_words + 0.0496 * average_sentence_length

            # if percent difficult words is about 5% then adjust score
            if percent_difficult_words > 5:
                grade += 3.6365
            return grade
        else:
            return 0

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

        if 'flesch_reading_ease' in self.features:
            data['features']['flesch_reading_ease'] = self.flesch_reading_ease(cnt_sentences, doc._.cnt_words, doc._.cnt_syllables)

        cnt_pos = Counter([token.pos_ for token in doc if token.pos_ in {
                    'NOUN', 'PROPN', 'ADJ', 'DET', 'ADP', 'PRON', 'VERB', 'AUX', 'ADV', 'INTJ'
                }])
        data['features']['heylighen_formality'] = self.get_heylighen_formality_score(cnt_pos)

        cnt_diff_words = 0
        for word in doc:
            if not word.is_punct and "'" not in word.text:
                if ( (word.text.lower() not in self.DALE_CHALL_WORDS) and (word.lemma_.lower() not in self.DALE_CHALL_WORDS) ):
                    cnt_diff_words += 1

        data['features']['dale_chall'] = self.dale_chall(cnt_sentences, doc._.cnt_words, cnt_diff_words)

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

        data['surface']['syllables'] = collections.Counter([i for g in [word._.syllables for word in (word for word in doc if not word.is_punct and "'" not in word.text and word._.syllables != None)] for i in g])
        data['surface']['letter_tokens'] = collections.Counter([token.text for token in doc if not token.is_punct and not token.is_digit])
        data['surface']['no_digit_tokens'] = collections.Counter([token.text for token in doc if not token.is_punct])
        data['surface']['sentences'] = collections.Counter([sent.text for sent in doc.sents])


        data['surface']['cnt_diff_words'] = cnt_diff_words

        data['counts']['toks_min_three_syllables'] = len(toks_min_three_syllables)
        data['counts']['toks_larger_six_letters'] = len(toks_larger_six_letters)
        data['counts']['toks_one_syllable'] = len(toks_one_syllable)

        data['counts']['syllables'] = sum(data['surface']['syllables'].values())
        data['counts']['letter_tokens'] = sum(data['surface']['letter_tokens'].values())
        data['counts']['no_digit_tokens'] = sum(data['surface']['no_digit_tokens'].values())

        return data

    def feat_corpus(self, corpus):
        """Featurize a corpus with surface level metrics

        Parameters
        ----------

        Returns
        -------
        dict
            SurfaceReadability level metrics of a document
        """

        data = {'features': {}}

        new_list = []
        for el in corpus.resources.surface['segments']:
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
                corpus.resources.surface['cnt_no_digit_tokens']
            )

        if 'forcast' in self.features:
            data['features']['forcast'] = self.forcast(
                corpus.resources.surface['cnt_words'],
                corpus_segments
            )

        if 'gunning_fog' in self.features:
            data['features']['gunning_fog'] = self.gunning_fog(
                corpus.sizes.sentences_cnt,
                corpus.resources.surface['cnt_words'],
                corpus.resources.surface['syllables_per_word']
            )

        if 'flesch_reading_ease' in self.features:
            data['features']['flesch_reading_ease'] = self.flesch_reading_ease(
                corpus.sizes.sentences_cnt,
                corpus.resources.surface['cnt_words'],
                corpus.resources.surface['cnt_syllables']
            )

        if 'heylighen_formality' in self.features:
            data['features']['heylighen_formality'] = self.get_heylighen_formality_score(corpus.resources.surface['cnt_pos'])

        if 'dale_chall' in self.features:
            data['features']['dale_chall'] = self.dale_chall(
                corpus.sizes.sentences_cnt,
                corpus.resources.surface['cnt_words'],
                corpus.resources.surface['cnt_diff_words']
            )

        return data
