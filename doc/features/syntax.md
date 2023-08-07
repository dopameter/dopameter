# Syntax

Here, metrics for dependency and constituency are provided:
* For dependency parsing, we use the parsers embedded in spaCy using common training models for different languages models.
* For constituency parsing we use the [Berkeley Neural Parser](https://pypi.org/project/benepar/).

* _Counts_ by the amount of node definitions aroused from the syntax tree graph structure.
  * _syntax_dependency_tree_: depends on language model, examples for German Language: 'nk', 'sb', 'ag', 'ROOT'; examples for English Language: 'acl', 'conj', 'xcomp'
  * _syntax_constituency_tree_: depends on language model, examples for 'de': 'ROOT', 'CAP', 'CS', 'VP'; examples for en: 'TOP', 'S', 'VP'

The following _Metrics_ take general parsed sentence graph properties into account and include:

* _MaxFan_ and _AvgFan_: the maximum fan-out of the parse tree, i.e., the highest number of directly dominated nodes from an ancestor node in the entire parse tree---and their averages
* _AvgMaxDepth_: average of maximum depth for each parse tree, i.e., the longest path from the root node to a leaf node,  
* _AvgOutdegreeCentralization_: average out-degree centralization value, computed over all dependency graphs of sentences, plus
* _AvgClosenessCentralization_: average closeness centralization of all dependency graphs of sentences, calculated as the reciprocal of the sum of the distance length (shortest paths) between two nodes $x$ and $y$ \cite{Bavelas1950CommunicationPI}.

We also provide metrics that apply dependency or constituency parse tree structures: 

* Dependency
  * _AvgDepDist_ and _MaxDepDist_: Scores the linear distance between two syntactically related words.

* Constituency
  * _AvgNonTerminales_sent_: Average amount of non-terminal symbols without leaves.% of constituency trees
  * _AvgConstituents_sent_ and _AvgLenConstituents_: Average amount of constituents per sentence, and their lengths.
  * _AvgTunits_sent_ and _AvgLenTunits_: Scores the main clause of a sentence, all subordinate clauses and non-clausal parts that are attached to or embedded in it---and their lengths.


* Configuration with defaults:


```jsonlines
  "tasks": ["features", "counts"],
  "features": {
    "syntax_dependency_metrics" : "default",
    "syntax_dependency_tree": "default",
    "syntax_constituency_metrics" : "default",
    "syntax_constituency_metrics": "default"
  }
```
* Configuration in detail:

```jsonlines
  "tasks": ["features", "counts"],
  "features": {
    "syntax_dependency_metrics" : [
                'AvgFan',
                'MaxFan',
                'AvgMaxDepth',
                'AvgDepDist',
                'MaxDepDist',
                'AvgOutdegreeCentralization',
                'AvgClosenessCentralization'
            ],
    "syntax_dependency_tree": ["nk", "sb", "ag"], # only three examples
    "syntax_constituency_metrics" : [
                'AvgMaxDepth',
                'AvgFan',
                'MaxFan',
                'AvgNonTerminales_sent',
                'AvgConstituents_sent',
                'AvgTunits_sent',
                'AvgLenConstituents',
                'AvgLenTunits',
                'AvgOutdegreeCentralization',
                'AvgClosenessCentralization'
            ],
    "syntax_constituency_tree": ["ROOT", "CAP", "CS", "VP"], # only three examples
  }
```

----
[Basic Counts (Token and Corpus Characteristics, POS, NER)](features/basics.md) | [N-Grams](features/ngrams.md) | [Lexical Richness](features/lexical_richness.md) | [Surface Patterns](features/surface.md) | [Syntax](features/syntax.md) | [Semantic Relations](features/semantic_relations.md) | [Emotion](features/emotion.md)
----
[Installation](../installation.md) | [Input & Data](../input.md) | [Functionality & Tasks](../tasks.md) | [Feature Hub](../features.md) | [Summarization](../analytics/summarization.md) | [Comparison](../analytics/comparison.md) | [Aggregation](../analytics/aggregation.md) | [Config & Run](../configuration.md)