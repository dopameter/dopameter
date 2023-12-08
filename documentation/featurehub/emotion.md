# Emotion

* DOPA METER supports scoring for emotions of the eight emotional variables (valence, arousal, dominance, joy, anger, sadness, fear and disgust) based on dictionary look-ups in the emotion lexicons from [MEmoLon](https://github.com/JULIELab/MEmoLon).
* Note / Warning: install the dictionaries of MEmoLon with the emotion scores.
  * Use [installation.py](../../install_languages.py) file for automatically download.
  * This process creates the directory `ext_res/emotion and stores `tsv` files for every needed language.

* All metrics of this package are computed document wise and corpus wise (micro and macro average).

* Configuration with defaults:

```jsonlines
  "tasks": ["features"],
  "features": {
    "emotion": "default"
  }
```
* Configuration in detail:

```jsonlines
  "tasks": ["features"],
  "features": {
    "emotion": [
      "valence",
      "arousal",
      "dominance",
      "joy",
      "anger",
      "sadness",
      "fear",
      "disgust"
    ]
  }
```

----
[Basic Counts (Token and Corpus Characteristics, POS, NER)](./basics.md) | [N-Grams](./ngrams.md) | [Lexical Diversity](./lexical_diversity.md) | [Surface Patterns](./surface.md) | [Syntax](./syntax.md) | [Semantic Relations](./semantic_relations.md) | [Emotion](features/emotion.md)

----
[Installation](../installation.md) | [Input & Data](../input.md) | [Functionality & Tasks](../tasks.md) | [Feature Hub](../features.md) | [Summarization](../analytics/summarization.md) | [Comparison](../analytics/comparison.md) | [Aggregation](../analytics/aggregation.md) | [Config & Run](../configuration.md)
