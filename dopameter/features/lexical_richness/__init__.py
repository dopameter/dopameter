import collections
import scipy
import math
import statistics
import numpy as np
import logging


class LexicalRichnessFeatures:
    """Get Lexical Richness Metrics

    Parameters
    ----------
    features : list, default=[
                "type_token_ratio",
                "guiraud_r",
                "herdan_c",
                "dugast_k",
                "maas_a2",
                "dugast_u",
                "tuldava_ln",
                "brunet_w",
                "cttr",
                "summer_s",
                "sttr",
                "sichel_s",
                "michea_m",
                "honore_h",
                "entropy",
                "yule_k",
                "simpson_d",
                "evenness",
                "mattr",
                "mtld"
                ]
    window_size = 30

    Attributes
    ----------
      * Metrics that use sample size and vocabulary size
        * `type_token_ratio`: Typical Type Token Ratio $\frac{V(N)}{N}$
        * `function words`: tokens with Part-of-Speech tag in ['VERB', 'AUX', 'ADJ', 'NOUN', 'PRON', 'ADJ'] (only as 'count', no 'metric')
        * `lexical density`: $\Sum_{function_words}*100$ / N
        * `guiraud_r`: $R = \frac{V(N)}{N}\$ (Guiraud, 1954)
        * `herdan_c`: $C = \frac{V(N)}{\sqrt{N}}$ (Herdan, 1960 & 1964)
        * `dugast_k`: $\frac{\log(V(N))}{\log(\log(N))}$ (Dugast, 1979)
        * `maas_a2`: $A² = \frac{\log(N) - \log(V(N))}{\log(N)^2}$ (Maas, 1972)
        * `dugast_u`: $U = \frac{\log(N)^2}{\log(N) - \log(V(N))}$ (Dugast, 1978 & 1979)
        * `tuldava_ln`: $LN = \frac{1 - V(N)^2}{V(N)^2\log(N)}$ (Tuldava, 1977)
        * `brunet_w`: $W = N^{V(N)^{-a}}; a = -0.172$ (Brunet, 1978)
        * `cttr`: $CTTR = \frac{V(N)}{\sqrt{2 N}}$ (Carroll's Corrected Type-Token Ratio, 1964)
        * `summer_s`: $\frac{\log(\log(V(N)))}{\log(\log(N))}$ (Summer's S index)
        * `sttr`: average standardized type-token ratio by a default window size (Kubat & Milicka, 2013)

      * Measures that use part of the frequency spectrum
        * `sichel_s`: $S = \frac{V(2, N)}{V(N)}$ (Sichel 1975)
        * `michea_m`: $M = \frac{V(N)}{V(2, N)}$ (Michéa 1969 & 1971)
        * `honore_h`: $H = 100 \frac{\log(N)}{1 - \frac{V(1, N)}{V(N)}}$ (Honoré 1979)

      * Measures that use the whole frequency spectrum
        * `entropy`: $\sum_{i=1}^N V(i, N)\left(-\log(\frac{i}{N})\right)\frac{i}{N}$ (Entropy)
        * `yule_k`: $K = 10^4 \left(-\frac{1}{N} + \sum_{i=1}^N V(i, N) \left( \frac{i}{N}\right)^2 \right)$ (Yule, 1944)
        * `simpson_d`: $D = \sum_{i=1}^{V(N)} V(i, N) \frac{i}{N} \frac{i - 1}{N - 1}$ (Simpson, 1949)
        * `herdan_vm`: $V_m = \sqrt{-\frac{1}{V(N)} + \sum_{i=1}^{V(N)} V(i, N) \left(\frac{i}{N}\right)^2}$ (Herdan, 1955)
        * `hdd`: $HD-D = \sum_{i=1}^{V(N)} \frac{1}{42} \left(1 - \frac{\binom{i}{0} \binom{N - V(i, N)}{42 - 0}}{\binom{N}{42}}\right) = \sum_{i=1}^{V(N)} \frac{1}{42} (1 - \frac{\binom{N - V(i, N)}{42}}{\binom{N}{42}})$" (McCarthy and Jarvis, 2010, see https://link.springer.com/content/pdf/10.3758/BRM.42.2.381.pdf)
        * `evenness`: derivated from Pielou's Evenness
        * `mattr`: Moving-Average Type-Token Ratio (Covington and McFall, 2010)
        * `mtld`: McCarthy and Jarvis (2010) measure of textual lexical diversity, see https://link.springer.com/content/pdf/10.3758/BRM.42.2.381.pdf


    Notes
    -----

    This package is derived from Thomas Proisl's textcomplexity package
    * https://pypi.org/project/textcomplexity/
    * https://github.com/tsproisl/textcomplexity/

    All here named measurements adapted from:
    https://github.com/tsproisl/textcomplexity/blob/master/doc/measures.pdf

    define in configuration .json file under features:

    * default
      "tasks": ["features", "counts"],
        "features": {
          "lexical_richness": "default"
        }
    * or in detail:
      "tasks": ["features", "counts"],
        "features": {
          "lexical_richness": [
          "type_token_ratio",
          "lexical_density",
          "guiraud_r",
          "herdan_c",
          "dugast_k",
          "maas_a2",
          "dugast_u",
          "tuldava_ln",
          "brunet_w",
          "cttr",
          "summer_s",
          "sttr",
          "sichel_s",
          "michea_m",
          "honore_h",
          "entropy",
          "yule_k",
          "simpson_d",
          "herdan_vm",
          "hdd",
          "evenness",
          "mattr",
          "mtld"
        ]
     }
    """

    def __init__(
            self,
            features='default',
            window_size=30
    ):

        logging.info('\tInitialize lexical richness features.')

        default_features = [
            'type_token_ratio',
            'lexical_density',
            'guiraud_r',
            'herdan_c',
            'dugast_k',
            'maas_a2',
            'dugast_u',
            'tuldava_ln',
            'brunet_w',
            'cttr',
            'summer_s',
            'sttr',
            'sichel_s',
            'michea_m',
            'honore_h',
            'entropy',
            'yule_k',
            'simpson_d',
            'herdan_vm',
            'hdd',
            'evenness',
            'mattr',
            'mtld'
        ]

        if features == 'default':
            self.features = default_features
        else:
            if set(self.features).intersection(default_features) == set():
                raise ValueError('Your lexical richness features ' + ' '.join(self.features) + ' are not defined! Allowed definitions: ', default_features)
            else:
                logging.info('\t\tDefined features: ' + features)
                self.features = features

        self.window_size = window_size

    def type_token_ratio(self, types, tokens):
        """Typical Type Token Ratio $\frac{V(N)}{N}$"""
        if tokens != 0:
            return types / tokens
        else:
            return 0

    def lexical_density(self, function_words, tokens):
        """Lexical density: $(function_words / N_tokens) * 100$"""
        if tokens != 0:
            return (len(function_words) * 100) / tokens
        else:
            return 0

    def guiraud_r(self, types, tokens):
        """Guiraud R (1954): $\frac{V(N)}{N}\$"""
        div = math.sqrt(tokens)
        if div != 0:
            return types / div
        else:
            return 0

    def herdan_c(self, types, tokens):
        """Herdan C (1960, 1964): $\frac{V(N)}{\sqrt{N}}$"""

        if tokens > 0:
            div = math.log(tokens)
            if div != 0:
                return math.log(types) / div
            else:
                return 0
        else:
            return 0

    def dugast_k(self, types, tokens):
        """Dugast (1979): $\frac{\log(V(N))}{\log(\log(N))}$"""

        if tokens > 1:
            div = math.log(math.log(tokens))
            if div != 0:
                return math.log(types) / div
            else:
                return 0
        else:
            return 0

    def maas_a2(self, types, tokens):
        """Maas (1972): $\frac{\log(N) - \log(V(N))}{\log(N)^2}$"""

        if tokens > 0:
            div = (math.log(tokens) ** 2)
            if div != 0:
                return math.log(tokens) - math.log(types) / div
            else:
                return 0
        else:
            return 0

    def dugast_u2(self, types, tokens):
        """Dugast (1978, 1979): $\frac{\log(N)^2}{\log(N) - \log(V(N))}$"""

        if tokens != 0 and types > 1:
            types = types - 1
            div = (math.log(tokens) - math.log(types))
            if div != 0:
                if tokens == types:
                    return (math.log(tokens) ** 2) / div
                else:
                    return (math.log(tokens) ** 2) / div
            else:
                return 0
        else:
            return 0

    def tuldava_ln(self, types, tokens):
        """Tuldava (1977): $LN = \frac{1 - V(N)^2}{V(N)^2\log(N)}$"""

        if tokens > 0:
            div = ((types ** 2) * math.log(tokens))
            if div != 0:
                return (1 - (types ** 2)) / div
            else:
                return 0
        else:
            return 0

    def brunet_w(self, types, tokens, a):
        """Brunet (1978): $W = N^{V(N)^{-a}}; a = -0.172$"""
        if tokens and types:
            return tokens ** (types ** -a)
        else:
            return 0

    def cttr(self, types, tokens):
        """Carroll's Corrected Type-Token Ratio: CTTR = \frac{V(N)}{\sqrt{2 N}"""

        div = math.sqrt(2 * tokens)
        if div != 0:
            return types / div
        else:
            return 0

    def summer_s(self, types, tokens):
        """Summer's S index: \frac{\log(\log(V(N)))}{\log(\log(N))}"""

        if types != 1 and tokens > 1:
            div = math.log(math.log(tokens))
            if div != 0:
                return math.log(math.log(types)) / div
            else:
                return 0
        else:
            return 0

    def sttr(self, tokens, doc_tokens, window_size):
        """average standardized type-token ratio by a default window size"""
        sttr = []
        for i in range(int(tokens / window_size)):  # ignore last partial chunk
            doc_segment = doc_tokens[i * window_size:(i * window_size) + window_size]
            text_length = len(doc_segment)
            vocab_length = len(set(token for token in doc_segment))
            sttr.append(self.type_token_ratio(text_length, vocab_length))
        if sttr:
            return np.mean(sttr)
        else:
            return 0

    def sichel_s(self, freq_spectrum, types):
        """Sichel (1975): $S = \frac{V(2, N)}{V(N)}$"""
        if types:
            return freq_spectrum.get(2, 0) / types
        else:
            return 0

    def michea_m(self, types, freq_spectrum):
        """Michéa (1969, 1971): $M = \frac{V(N)}{V(2, N)}$"""

        spectrum = freq_spectrum.get(2, 0)
        if spectrum:
            return types / spectrum
        else:
            return 0

    def honore_h(self, tokens, types, freq_spectrum):
        """Honoré (1979): $H = 100 \frac{\log(N)}{1 - \frac{V(1, N)}{V(N)}}$"""
        hapaxes = freq_spectrum.get(1, 0)
        if hapaxes == types:
            hapaxes -= 1

        if types != 0:
            value = 1 - (hapaxes / types)
            if value != 0:
                return 100 * math.log(tokens) / value
            else:
                return 0
        else:
            return 0

    def entropy(self, tokens, freq_spectrum):
        """Calculate Entropy: $\sum_{i=1}^N V(i, N)\left(-\log(\frac{i}{N})\right)\frac{i}{N}$"""
        if tokens != 0:
            return sum((freq_size * (- math.log(freq / tokens)) * (freq / tokens) for freq, freq_size in freq_spectrum.items()))
        else:
            return 0

    def yule_k(self, tokens, freq_spectrum):
        """Yule (1944): $K = 10^4 \left(-\frac{1}{N} + \sum_{i=1}^N V(i, N) \left( \frac{i}{N}\right)^2 \right)$"""
        if tokens != 0:
            return 100 * (sum((freq_size * (freq / tokens) ** 2 for freq, freq_size in freq_spectrum.items())) - (1 / tokens))
        else:
            return 0

    def simpson_d(self, tokens, freq_spectrum):
        """Simpson (1949): $D = \sum_{i=1}^{V(N)} V(i, N) \frac{i}{N} \frac{i - 1}{N - 1}$"""
        if tokens > 1:
            return sum((freq_size * (freq / tokens) * ((freq - 1) / (tokens - 1)) for freq, freq_size in freq_spectrum.items()))
        else:
            return 0

    def herdan_vm(self, tokens, types, freq_spectrum):
        """Herdan (1955) $V_m = \sqrt{-\frac{1}{V(N)} + \sum_{i=1}^{V(N)} V(i, N) \left(\frac{i}{N}\right)^2}$"""
        if tokens != 0 and types != 0:
            herdan_vm_sum = sum((freq_size * (freq / tokens) ** 2 for freq, freq_size in freq_spectrum.items()))
            herdan_vm_frac = (1 / types)
            h_sum = herdan_vm_sum - herdan_vm_frac
            if herdan_vm_sum - herdan_vm_frac > 0:
                return math.sqrt(h_sum)
            else:
                return 0
        else:
            return 0

    def hdd(self, tokens, frequency_spectrum, sample_size=42):
        """McCarthy and Jarvis (2010): $HD-D = \sum_{i=1}^{V(N)} \frac{1}{42} \left(1 - \frac{\binom{i}{0} \binom{N - V(i, N)}{42 - 0}}{\binom{N}{42}}\right) = \sum_{i=1}^{V(N)} \frac{1}{42} (1 - \frac{\binom{N - V(i, N)}{42}}{\binom{N}{42}})$"""
        if sample_size > 0:
            return sum((((1 - scipy.stats.hypergeom.pmf(0, tokens, freq, sample_size)) / sample_size) for word, freq in frequency_spectrum.items()))
        else:
            return 0

    def evenness(self, entropy, freq_spectrum):
        """derivated from Pielou's Evenness"""

        if len(freq_spectrum) > 0:
            math_log_len_spec = math.log(len(freq_spectrum))
            if math_log_len_spec != 0:
                return entropy / math_log_len_spec
            else:
                return 0
        else:
            return 0

    def mattr(self, window_size, doc_tokens, n_tokens):
        """Moving-Average Type-Token Ratio (Covington and McFall, 2010).
            M.A. Covington, J.D. McFall: Cutting the Gordon Knot. In: Journal
            of Quantitative Linguistics 17,2 (2010), p. 94-100. DOI:
            10.1080/09296171003643098
        """

        ttr_values = []
        window_frequencies = collections.Counter(doc_tokens[0:window_size])
        for window_start in range(1, n_tokens - (window_size + 1)):
            window_end = window_start + window_size
            word_to_pop = doc_tokens[window_start - 1]
            window_frequencies[word_to_pop] -= 1
            window_frequencies[doc_tokens[window_end]] += 1
            if window_frequencies[word_to_pop] == 0:
                del window_frequencies[word_to_pop]
            # type-token ratio for the current window:
            ttr_values.append(len(window_frequencies) / window_size)
        if not ttr_values:
            return 0
        else:
            return statistics.mean(ttr_values)

    def fct_mtld(self, doc_tokens, n_tokens, factor_size, reverse=False):
        """Function needed in McCarthy and Jarvis (2010)."""

        factors = 0
        types = set()
        token_count = 0
        token_iterator = iter(doc_tokens)
        if reverse:
            token_iterator = reversed(doc_tokens)
        for token in token_iterator:
            types.add(token)
            token_count += 1
            if len(types) / token_count <= factor_size:
                factors += 1
                types = set()
                token_count = 0
        if token_count > 0:
            ttr = len(types) / token_count
            factors += (1 - ttr) / (1 - factor_size)

        if factors == 0:
            return 0
        else:
            return n_tokens / factors

    def mtld(self, doc_tokens, n_tokens, factor_size):
        """McCarthy and Jarvis (2010) measure of textual lexical diversity,
        see https://link.springer.com/content/pdf/10.3758/BRM.42.2.381.pdf"""
        forward_mtld = self.fct_mtld(doc_tokens, n_tokens, factor_size, reverse=False)
        reverse_mtld = self.fct_mtld(doc_tokens, n_tokens, factor_size, reverse=True)
        return statistics.mean((forward_mtld, reverse_mtld))

    def feat_doc(self, doc):
        """Get features for doc

        Parameters
        ----------
        doc : spaCy Doc

        Returns
        -------
        dict
            dictionary with 'features' as key including the metrics and 'lexical_richness' with interim results and 'counts' including amount of function words
        """

        data = {'features': {}, 'counts': {}, 'lexical_richness': {}}
        freq_list = collections.Counter(doc._.tokens)
        freq_spectrum = dict(collections.Counter(freq_list.values()))

        # MEASURES THAT USE SAMPLE SIZE AND VOCABULARY SIZE #

        if 'type_token_ratio' in self.features:
            data['features']['type_token_ratio'] = self.type_token_ratio(doc._.n_vocab_types, doc._.n_tokens)

        function_words = [token.text for token in doc if token.pos_ if token.pos_ in ['VERB', 'AUX', 'ADJ', 'NOUN', 'PRON', 'ADJ']]
        if 'lexical_density' in self.features:
            data['features']['lexical_density'] = self.lexical_density(function_words, doc._.n_tokens)
            data['counts']['function_words'] = len(function_words)
            data['lexical_richness']['function_words'] = function_words

        if 'guiraud_r' in self.features:
            data['features']['guiraud_r'] = self.guiraud_r(doc._.n_vocab_types, doc._.n_tokens)

        if 'herdan_c' in self.features:
            data['features']['herdan_c'] = self.herdan_c(doc._.n_vocab_types, doc._.n_tokens)

        if 'herdan_c' in self.features:
            data['features']['herdan_c'] = self.dugast_k(doc._.n_vocab_types, doc._.n_tokens)

        if 'maas_a2' in self.features:
            data['features']['maas_a2'] = self.maas_a2(doc._.n_vocab_types, doc._.n_tokens)

        if 'maas_a2' in self.features:
            data['features']['maas_a2'] = self.dugast_u2(doc._.n_vocab_types, doc._.n_tokens)

        if 'tuldava_ln' in self.features:
            data['features']['tuldava_ln'] = self.tuldava_ln(doc._.n_vocab_types, doc._.n_tokens)

        if 'brunet_w' in self.features:
            data['features']['brunet_w'] = self.brunet_w(doc._.n_vocab_types, doc._.n_tokens, a=0.172)

        if 'cttr' in self.features:
            data['features']['cttr'] = self.cttr(doc._.n_vocab_types, doc._.n_tokens)

        if 'summer_s' in self.features:
            data['features']['summer_s'] = self.summer_s(doc._.n_vocab_types, doc._.n_tokens)

        if 'sttr' in self.features:
            data['features']['sttr'] = self.sttr(doc._.n_vocab_types, doc._.tokens, self.window_size)

        # MEASURES THAT USE PART OF THE FREQUENCY SPECTRUM #

        if 'sichel_s' in self.features:
            data['features']['sichel_s'] = self.sichel_s(freq_spectrum, doc._.n_vocab_types)

        if 'michea_m' in self.features:
            data['features']['michea_m'] = self.michea_m(doc._.n_vocab_types, freq_spectrum)

        if 'guiraud_r' in self.features:
            data['features']['michea_m'] = self.honore_h(doc._.n_tokens, doc._.n_vocab_types, freq_spectrum)

        # MEASURES THAT USE THE WHOLE FREQUENCY SPECTRUM #

        if 'entropy' in self.features:
            data['features']['entropy'] = self.entropy(doc._.n_vocab_types, freq_spectrum)

        if 'yule_k' in self.features:
            data['features']['yule_k'] = self.yule_k(doc._.n_vocab_types, freq_spectrum)

        if 'simpson_d' in self.features:
            data['features']['simpson_d'] = self.simpson_d(doc._.n_vocab_types, freq_spectrum)

        if 'herdan_vm' in self.features:
            data['features']['herdan_vm'] = self.herdan_vm(doc._.n_tokens, doc._.n_vocab_types, freq_spectrum)

        if 'hdd' in self.features:
            data['features']['hdd'] = self.hdd(doc._.n_tokens, freq_spectrum, sample_size=42)

        if 'evenness' in self.features:
            data['features']['evenness'] = self.evenness(data['features']['entropy'], freq_spectrum)

        if 'mattr' in self.features:
            data['features']['mattr'] = self.mattr(self.window_size, doc._.tokens, doc._.n_tokens)

        if 'mtld' in self.features:
            data['features']['mtld'] = self.mtld(doc._.tokens, doc._.n_tokens, factor_size=0.72)

        data['lexical_richness']['freq_list'] = freq_list
        data['lexical_richness']['corpus_tokens'] = doc._.tokens

        return data

    def feat_corpus(self, corpus):
        """Get lexical_richness features for corpus

        Parameters
        ----------
        corpus: corpus

        Returns
        -------
        dict
            dictionary with 'features' as key including the metrics
        """

        n_types = len(corpus.types)

        data = {'features': {}}
        freq_spectrum = dict(collections.Counter(corpus.lexical_richness['freq_list'].values()))

        # MEASURES THAT USE SAMPLE SIZE AND VOCABULARY SIZE #

        if 'type_token_ratio' in self.features:
            data['type_token_ratio'] = self.type_token_ratio(n_types, corpus.tokens_cnt)

        if 'lexical_density' in self.features:
            data['lexical_density'] = self.lexical_density(corpus.lexical_richness['function_words'], corpus.tokens_cnt)

        if 'guiraud_r' in self.features:
            data['guiraud_r'] = self.guiraud_r(n_types, corpus.tokens_cnt)

        if 'herdan_c' in self.features:
            data['herdan_c'] = self.herdan_c(n_types, corpus.tokens_cnt)

        if 'dugast_k' in self.features:
            data['dugast_k'] = self.dugast_k(n_types, corpus.tokens_cnt)

        if 'maas_a2' in self.features:
            data['maas_a2'] = self.maas_a2(n_types, corpus.tokens_cnt)

        if 'dugast_u' in self.features:
            data['dugast_u'] = self.dugast_u2(n_types, corpus.tokens_cnt)

        if 'tuldava_ln' in self.features:
            data['tuldava_ln'] = self.tuldava_ln(n_types, corpus.tokens_cnt)

        if 'brunet_w' in self.features:
            data['brunet_w'] = self.brunet_w(n_types, corpus.tokens_cnt, a=0.172)

        if 'cttr' in self.features:
            data['cttr'] = self.cttr(n_types, corpus.tokens_cnt)

        if 'summer_s' in self.features:
            data['summer_s'] = self.summer_s(n_types, corpus.tokens_cnt)

        if 'sttr' in self.features:
            data['sttr'] = self.sttr(n_types, corpus.lexical_richness['corpus_tokens'], self.window_size)

        # MEASURES THAT USE PART OF THE FREQUENCY SPECTRUM #

        if 'sichel_s' in self.features:
            data['sichel_s'] = self.sichel_s(freq_spectrum, n_types)

        if 'michea_m' in self.features:
            data['michea_m'] = self.michea_m(n_types, freq_spectrum)

        if 'honore_h' in self.features:
            data['honore_h'] = self.honore_h(corpus.tokens_cnt, n_types, freq_spectrum)

        # MEASURES THAT USE THE WHOLE FREQUENCY SPECTRUM #

        if 'entropy' in self.features:
            data['entropy'] = self.entropy(n_types, freq_spectrum)

        if 'yule_k' in self.features:
            data['yule_k'] = self.yule_k(n_types, freq_spectrum)

        if 'simpson_d' in self.features:
            data['simpson_d'] = self.simpson_d(n_types, freq_spectrum)

        if 'herdan_vm' in self.features:
            data['herdan_vm'] = self.herdan_vm(corpus.tokens_cnt, n_types, freq_spectrum)

        if 'hdd' in self.features:
            data['hdd'] = self.hdd(corpus.tokens_cnt, freq_spectrum, sample_size=42)

        if 'evenness' in self.features:
            data['evenness'] = self.evenness(data['entropy'], freq_spectrum)

        if 'mattr' in self.features:
            data['mattr'] = self.mattr(
                self.window_size,
                corpus.lexical_richness['corpus_tokens'],
                corpus.tokens_cnt
            )

        if 'mtld' in self.features:
            data['mtld'] = self.mtld(
                corpus.lexical_richness['corpus_tokens'],
                corpus.tokens_cnt,
                factor_size=0.72
            )

        return {'features': {f: data[f] for f in self.features}}
