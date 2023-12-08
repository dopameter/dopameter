# Semantic Relations

We investigate topological properties of lexicons, in particular the linkage of lemmas by various semantic relations.
Lexicons structured this way can be regarded as semantic networks.
For relation scoring we used the [Python package wn](https://github.com/goodmami/wn), which provides WordNets for various languages,
such as the [Open English WordNet](https://github.com/globalwordnet/english-wordnet) and the [OdeNet](https://github.com/hdaSprachtechnologie/odenet).

We provide a set of features that is defined by the identifiers if the graph nodes:
for WordNet _synsets_ `wordnet_synsets` such as _senses_ `wordnet_senses` is provided.
Both identifiers based features can be used as input for the compare modus.

DOPA METER contains several _metrics_ that exploit the topological structure of these semantic networks for lexicon entries (lemmas) within each sentence: 
* `sem_depth_min`: minimal path length of each reading of each lemma within in a document (distance from the top node of the semantic network to the lemma) following taxonomic links (hypernym or hyponymy links, only), sum up these individual length scores and average over the number of all the lemmas' readings.
* `sem_depth_max: maximal path length of each reading of each lemma within in a document (distance from the top node of the semantic network to the lemma) following taxonomic links (hypernym or hyponymy links, only), sum up these individual length scores and average over the number of all the lemmas' readings.
* `sem_rich_*`: For each (reading of the) lemma in a sentence, we determine all of its semantic relation instances (i.e., hypernyms, hyponyms, parts (is-part) and wholes (has-part), antonyms) it shares with other lemmas in the lexicon and average this number over all readings per lemma and all sentences in the document. We also supply semantic richness scores for each specific semantic relation (i.e., sem_rich_hypernyms, ..., sem_rich_antonyms).
* `synsets_avg` : average amount of synsets in a document / corpus
* `senses_avg` : average amount of senses in a document / corpus

* Configuration with defaults:

```jsonlines
   "tasks": ["features", "counts"],
   "features": {
     "wordnet_synsets": "default",
     "wordnet_senses": "default",
     "wordnet_semantic_relations": "default"
   }
```

* Configuration in detail:

```jsonlines
  "tasks": ["features", "counts"],
  "features": {
     "wordnet_semantic_relations":
        [
          'sem_rich_hypernyms',
          'sem_rich_hyponyms',
          'sem_rich_taxonyms',
          'sem_rich_antonyms',
          'sem_rich_synonyms',
          'sem_rich_meronyms',
          'sem_rich_holonyms',
          'sem_rich',
          'sem_depth_min',
          'sem_depth_max',
          'synsets_avg',
          'senses_avg'
        ],
     "wordnet_synsets": ["odenet-11460-n", "odenet-3279-a, "odenet-15368-a"], # only examples, for the full list, check the docuemntation of GermaNet
     "wordnet_senses": ["w45444_11460-n", "w15016_3279-a", "w15016_15368-a"], # only examples, for the full list, check the docuemntation of GermaNet
    }
```

## Dictionary Lookup

_Dictionary Lookup_ provides a set of metrics, where _Counts_ and _Metrics_ are scored by simple look-up of self configurable dictionaries.


## Configuration

```jsonlines
  "features": {
    "dictionary_lookup": ["examples_1", "examples_2"]
  },
  "external_resources":
  {
    "dictionaries":
    {
      "examples_1": "/example_dictionaries/dicts_1",
      "examples_2": "/example_dictionaries/dicts_2"
    }
  }
```

----
[Basic Counts (Token and Corpus Characteristics, POS, NER)](./basics.md) | [N-Grams](./ngrams.md) | [Lexical Diversity](./lexical_diversity.md) | [Surface Patterns](./surface.md) | [Syntax](./syntax.md) | [Semantic Relations](./semantic_relations.md) | [Emotion](features/emotion.md)

----
[Installation](../installation.md) | [Input & Data](../input.md) | [Functionality & Tasks](../tasks.md) | [Feature Hub](../features.md) | [Summarization](../analytics/summarization.md) | [Comparison](../analytics/comparison.md) | [Aggregation](../analytics/aggregation.md) | [Config & Run](../configuration.md)