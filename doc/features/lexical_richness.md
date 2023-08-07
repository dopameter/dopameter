# Lexical Richness

* Subsumes feature set which addresses vocabulary metrics that mainly come up in the field of stylometrics
* For a survey of metrics of lexical diversity, cf.
  * David Malvern, Brian Richards, Ngoni Chipere, and Pilar Durán. 2004. [Lexical Diversity and Language Development: Quantification and Assessment](https://link.springer.com/book/10.1057/9780230511804). Palgrave Macmillan.
  * Stefan Evert, Peter Uhrig, Sabine Bartsch, and Thomas Proisl. 2017. [E-VIEW-alation – a large-scale evaluation study of association measures for collocation identification](https://elex.link/elex2017/wp-content/uploads/2017/09/paper32.pdf). In Electronic lexicography in the 21st century.  Proceedings of the eLex 2017 conference, pages 531–549, Leiden. Lexical Computing.

* The package `lexical_richness is derived from
  * https://pypi.org/project/textcomplexity/
  * https://github.com/tsproisl/textcomplexity/
  * in Detail: https://github.com/tsproisl/textcomplexity/blob/master/textcomplexity/vocabulary_richness.py

* All metrics of this package are computed document wise and corpus wise (micro and macro average).

* Features in details:

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


* Configuration with defaults:

```jsonlines
  "tasks": ["features", "counts"],
  "features": {
    "lexical_richness": "default"
  }
```
* Configuration in detail:

```jsonlines
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
```

----
[Basic Counts (Token and Corpus Characteristics, POS, NER)](features/basics.md) | [N-Grams](features/ngrams.md) | [Lexical Richness](features/lexical_richness.md) | [Surface Patterns](features/surface.md) | [Syntax](features/syntax.md) | [Semantic Relations](features/semantic_relations.md) | [Emotion](features/emotion.md)
----
[Installation](../installation.md) | [Input & Data](../input.md) | [Functionality & Tasks](../tasks.md) | [Feature Hub](../features.md) | [Summarization](../analytics/summarization.md) | [Comparison](../analytics/comparison.md) | [Aggregation](../analytics/aggregation.md) | [Config & Run](../configuration.md)