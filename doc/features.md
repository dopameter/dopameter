# Features

The computation of features is derived in (1) the estimation of _Counts_, where only the occurrences of single properties focused and (2) _Metrics_, that are partially computed by _Counts_ in relation of document and corpus size. For this step, micro statistics (document level) and macro statistics (corpus level) statistics are supported.

The computation is configurable by a choose-able list for an individual mode or a default-mode that computes all features.

The output of the featureization process is stored under a configured directory `featutres` and `summary`.

For details of the features, look into the following detailed documentation:

* [Basic Counts ](features/basics.md)
  * Corpus / Doc Counts
  * Token Characteristics
  * Part of Speech
  * Named Entites
* [$n$-grams](features/ngrams.md)
* [Lexical Richness](features/lexical_richness.md)
* [Surface Patterns](features/surface.md)
* [Syntax](features/syntax.md)
* [Semantic Relations](features/semantic_relations.md)
* [Emotion](features/emotion.md)

----
[Installation](../installation.md) | [Input & Data](input.md) | [Functionality & Tasks](tasks.md) | [Feature Hub](features.md) | [Summarization](analytics/summarization.md) | [Comparison](analytics/comparison.md) | [Aggregation](analytics/aggregation.md) | [Config & Run](configuration.md)