# Input: Data preparation

* The input is a set of corpora.
* Every corpus is a directory consisting single files including plaintext (file ending is configurable).
* [Example set up of 2 corpora](res/example_corpora)
  * [gra](res/example_corpora/gra): 2 files from [GraSCCo corpus](https://zenodo.org/record/6539131)
  * [wiki](res/example_corpora/wiki): 2 files originated from [German Wikipedia](https://de.wikipedia.org/wiki/Wikipedia:Hauptseite)
* It is not necessary, to name the file ending by `.txt`, this is [configurable](c).
* Note: The limit of every input text file is limited by 1 000 000 characters, because there is a [limit by the used pipeline of spaCy](https://github.com/explosion/spaCy/issues/2508).

----
[Installation](../installation.md) | [Input & Data](input.md) | [Functionality & Tasks](tasks.md) | [Feature Hub](features.md) | [Summarization](analytics/summarization.md) | [Comparison](analytics/comparison.md) | [Aggregation](analytics/aggregation.md) | [Config & Run](configuration.md)