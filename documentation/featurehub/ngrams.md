# N-Grams
The feature set of N-Grams is build by groups of (configurable) N=\{1,2,3,...\} tokens.
The scores calculate the ratio of N-Grams by single documents and by whole corpora.
This feature set contains the frequency of the vocabulary---known as the _term frequency–inverse document frequency (tf-idf)_.
Configure N, `most_frequent_words` as threshold, preferred 2000.
If `"store_sources": true`, the n-gram statistics of the whole corpora are stored in json files in your configured _output section_ `"path_sources": "/home/chlor/data/output_feattext/fix/sources"`.

It is possible to compute N-Grams of Part-of-Speech tokens. Configure `"mode_ngrams"`: `pos` in the settings sections.
This additional mode works for `pos` - only. It prepared for further development.

* All metrics of this package are computed document wise and corpus wise (micro and macro average).

* Configuration
```jsonlines
  "settings": {
    "store_sources": true, # or false
    "most_frequent_words": 2000
  },
  "output": {
      "path_features": "/home/chlor/data/output_feattext/usecase/features",
      "path_summary":  "/home/chlor/data/output_feattext/usecase/summary",
      "path_sources":  "/home/chlor/data/output_feattext/usecase/sources",
    },
  "features": {
    "ngrams": [1,2,3]
  }
```

----
[Basic Counts (Token and Corpus Characteristics, POS, NER)](./basics.md) | [N-Grams](./ngrams.md) | [Lexical Diversity](./lexical_diversity.md) | [Surface Patterns](./surface.md) | [Syntax](./syntax.md) | [Semantic Relations](./semantic_relations.md) | [Emotion](features/emotion.md)

----
[Installation](../installation.md) | [Input & Data](../input.md) | [Functionality & Tasks](../tasks.md) | [Feature Hub](../features.md) | [Summarization](../analytics/summarization.md) | [Comparison](../analytics/comparison.md) | [Aggregation](../analytics/aggregation.md) | [Config & Run](../configuration.md)
