# Input: Data preparation

* The input is a set of corpora.
* Every corpus is a directory consisting single files with plaintext (file ending is configurable).
* Corpora can be subsumed into collections via [configuration](configuration.md).
* [Example set up of 2 corpora](resources/example_corpora)
  * [gra](resources/example_corpora/gra): 2 files from [GraSCCo corpus](https://zenodo.org/record/6539131)
  * [wiki](resources/example_corpora/wiki): 2 files originated from [German Wikipedia](https://de.wikipedia.org/wiki/Wikipedia:Hauptseite)
* Use case from [EMNLP Demo 2023](https://aclanthology.org/2023.emnlp-demo.18/) setting: [https://zenodo.org/doi/10.5281/zenodo.10000770](https://zenodo.org/doi/10.5281/zenodo.10000770)
  * 20 corpora
  * 6 collections
  * 2 languages
* It is not necessary, to name the file ending by `.txt`, this is [configurable](configuration.md).
* Note: The limit of every input text file is limited by 1 000 000 characters, because there is a [limit by the used pipeline of spaCy](https://github.com/explosion/spaCy/issues/2508).

----
[Installation](installation.md) | [Input & Data](input.md) | [Functionality & Tasks](tasks.md) | [Feature Hub](features.md) | [Summarization](analytics/summarization.md) | [Comparison](analytics/comparison.md) | [Aggregation](analytics/aggregation.md) | [Config & Run](configuration.md)