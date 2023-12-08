# Features

The computation of features and metrics is derived in

1. the estimation of _Counts_, where only the occurrences of single properties focused and
2. _Metrics_, that are partially computed by _Counts_ in relation of document and corpus size.
For this step, micro statistics (document level) and macro statistics (corpus level) statistics are supported.

The computation is configurable by a choose-able list for an individual mode or a default-mode that computes all features.

The output of this process is stored under a configured directory `featutres` and `summary`.

For details of the metrics, look into the following detailed documentation:

* [Basic Counts ](featurehub/basics.md)
  * [Corpus / Doc Counts](featurehub/basics.md)
  * [Token Characteristics](featurehub/basics.md)
  * [Part of Speech](featurehub/basics.md)
  * [Named Entities](featurehub/basics.md)
* [_N_-grams](featurehub/ngrams.md)
* [Lexical Diversity](featurehub/lexical_diversity.md)
* [Surface Patterns](featurehub/surface.md)
* [Syntax](featurehub/syntax.md)
* [Semantic Relations](featurehub/semantic_relations.md)
* [Emotion](featurehub/emotion.md)

----
[Installation](installation.md) | [Input & Data](input.md) | [Functionality & Tasks](tasks.md) | [Feature Hub](features.md) | [Summarization](analytics/summarization.md) | [Comparison](analytics/comparison.md) | [Aggregation](analytics/aggregation.md) | [Config & Run](configuration.md)