# Basics: Basics: Corpus Characteristics, Token Characteristics, Part of Speech, Named Entities

The simplest way to describe a text corpus is the amount of sentences such as different sentences, tokens, the vocabulary size in type, lemmata and the size of a document in characters.
This is subsumed by the feature set `Corpus Characteristics`. This is the only feature, that you configure in `tasks`.
_Token Characteristics_, e.g., `is_alpha`, `is_numeric` (in detail, see spaCy documentation of [Linguistic Features](https://spacy.io/usage/linguistic-features) and [Token attributes](https://spacy.io/api/token#attributes))
the counts and deviations of _Part of Speech_ tags and the _Named Entities_ build three more basics feature sets and derived from spaCy embedded language models.

* Configuration with defaults:

```jsonlines
  "tasks": ["features", "counts", "corpus_characteristics"],
  "features": {
    "ner": "default",
    "pos": "default",
    "token_characteristics": "default",
  }
```
* Configuration in detail:

```jsonlines
  "tasks": ["features", "counts", "corpus_characteristics"],
  "features": {
    "ner": ["LOC", "ORG", "PERS", "MISC"],
    "pos": ["NNP", "VBG", "NN", "$"],
    "token_characteristics": ["is_alpha", "is_ascii", "is_digit", "is_lower", "is_upper", "is_title", "is_punct",
                             "is_left_punct", "is_right_punct", "is_space", "is_bracket", "is_quote", "is_currency",
                             "like_url", "like_num", "like_email", "is_oov", "is_stop"]
  }
```

----
[Basic Counts (Token and Corpus Characteristics, POS, NER)](features/basics.md) | [N-Grams](features/ngrams.md) | [Lexical Richness](features/lexical_richness.md) | [Surface Patterns](features/surface.md) | [Syntax](features/syntax.md) | [Semantic Relations](features/semantic_relations.md) | [Emotion](features/emotion.md)
----
[Installation](../installation.md) | [Input & Data](../input.md) | [Functionality & Tasks](../tasks.md) | [Feature Hub](../features.md) | [Summarization](../analytics/summarization.md) | [Comparison](../analytics/comparison.md) | [Aggregation](../analytics/aggregation.md) | [Config & Run](../configuration.md)