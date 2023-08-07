# Compare Analysis

DoPa Meters Comparison mode contrast differences between text corpora by absolute and portion-wise intersection and difference analysis of the vocabulary, $n$-grams and scores based on the _Counts_ of the feature set of Semantic Relations, e.g., synsets.

* **Mandatory:**
  1. Run of `'features'` before or in same run
  2. Store resources (interim results from features) during running `'features'`
    * Configuration in settings:
      * `"store_sources": true`
      * `"path_compare": "path/sources"`
    * [Example sources / interim results of one corpus](../res/results/sources/wiki)

* **Decentralized approach:**
  * You can run the features from the task `'features'` on different places and store the interim results via `"store_sources": true` in the configuration. 
  * If you want to run the compare mode of corpora from different places, copy the path with the interim results from _different_ `"path_compare"` into one directory `"path_compare"` and define it into one [configuration file](../res/example_configurations).    

* It computes intersections and differences of corpora (csv files) and visualize it in heatmaps (`plot mode):
  * Intersection of items, of the frequency of item and both portion-wise
    * [Example: intersection of 1-grams (vocabular)](../res/results/compare/ngrams_1/ngrams_1_intersection.csv)
      ![arc](../res/results/compare/ngrams_1/ngrams_1_intersection.png)
  * Difference of items, of the frequency of item and both portion-wise, example:
    * [Example: difference of 1-grams (vocabular)](../res/results/compare/ngrams_1/ngrams_1_difference.csv)
      ![arc](../res/results/compare/ngrams_1/ngrams_1_difference.png)
  * Intersections and differences stored in json-files.
    * [Example intersection](../res/results/compare/ngrams_1/ngrams_1_intersection.json) 
    * [Example difference](../res/results/compare/ngrams_1/ngrams_1_difference.json) 
* Metrics from language generation and language translation (used from NLTK):
  * [BLEU](https://www.nltk.org/_modules/nltk/translate/bleu_score.html)
  * [METEOR](https://www.nltk.org/howto/meteor.html)
  * [NIST](https://www.nltk.org/_modules/nltk/translate/nist_score.html)
    * Distances metrics (only via vocabulary / $1$-grams):
      * _Burrows' Delta_
      * _Manhattan Distance_
      * _Euclidean Distance_
      * _Squared Euclidean Distance_
      * _Cosine Distance_
      * _Canberra Distance_
      * _Bray-Curtis Distance_
      * _Correlation Distance_
      * _Chebyshev Distance_
      * _Quadratic Delta_
      * _Eder's Delta_
      * _Cosine Delta_
    * The implementation is derived from [PyDelta](https://github.com/cophi-wue/pydelta).


## Configuration

```jsonlines
    "compare": [
      "portion_intersection",
      "portion_intersection_all_occurrences",
      "portion_difference",
      "portion_difference_all_occurrences",
      "bleu",
      "meteor",
      "nist",
      "burrows",
      "manhattan",
      "euclidean",
      "sqeuclidean",
      "cosine",
      "canberra",
      "braycurtis",
      "correlation",
      "chebyshev",
      "quadratic",
      "eder",
      "cosine_delta"
    ]
```
----
[Installation](../installation.md) | [Input & Data](../input.md) | [Functionality & Tasks](../tasks.md) | [Feature Hub](../features.md) | [Summarization](../analytics/summarization.md) | [Comparison](../analytics/comparison.md) | [Aggregation](../analytics/aggregation.md) | [Config & Run](../configuration.md)