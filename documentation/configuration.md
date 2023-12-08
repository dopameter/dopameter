# Configure your `config.json`

(Mandatory definitions are highlighted in bold)

## Input: Corpora of Text Files

* `corpora`: defines a set of corpora ([examples](resources/example_corpora))
* Every corpus is defined by the name of the corpus (e.g., `Gra`, `Wiki` ) and has to be defined by.
  * **`path_text_data`**: Input path with plain text files, it's mandatory.
  * **`language`**: Language of corpus, it's mandatory.
  * `file_formats`: List of file format definitions, more definitions are possible; if your input files have no file ending define `[""]`; it's not mandatory, if there is no specified format, files with `.txt` file formats are used.
  * `encoding`: Define the encoding of the files, default is `'utf-8'`.
  * `collection`: Name of the collection, where the corpus belongs, it's not mandatory.
* Example configuration of two corpora:
    ```jsonlines
    {
      "corpora": {
          "Gra": {
          "path_text_data": "/path/to/gra/",
          "language": "de",
          "file_formats": ["txt"]
          "encoding": 'utf-8'
        },
        "Wiki": {
          "path_text_data": "/path/to/wiki/",
          "language": "de"
        },
      ...
    ```

### Example files
* Setting of 2 very small corpora: [resources/](resources/example_corpora)
* Use case from [EMNLP Demo 2023](https://aclanthology.org/2023.emnlp-demo.18/) setting: [https://zenodo.org/doi/10.5281/zenodo.10000770](https://zenodo.org/doi/10.5281/zenodo.10000770)
  * 20 corpora
  * 6 collections
  * 2 languages

## Common Settings

* The section of `settings` in the configuration files defines common definitions used in the featurehub and output tasks.
  * `"tasks"` defines a list of defined tasks, what feattext should during a run, description below.
    * possible values: `features`, `counts`, `corpus_characteristics`, `features`, `features_detail`, `plot`, `counts`, `corpus_characteristics`, `compare`, `store_sources` 
  * `"store_sources"`: a boolean value (`true` or `false`) defines if resources to compute comparing features are stored on the main memory
  * `"file_format_features"`: a list of file endings defines wich storage formats are used for the output of features, e.g.: `["csv", "excel"]`
  * `"file_format_clustering"`: `"csv"`
  * `"file_format_plots"`: a list of file endings defines wich storage formats are used for the output of plots, e.g.: `["png", "svg"]`,
  * `"file_format_dicts"`: `"txt"`
  * `"boxplot_height`: `5`- High of box plots created by _features detail_
  * `"most_frequent_words"`: `2000` - Most frequent words, used for n-grams / tfidf and distance metrics in compare mode.
  * `"mode_ngrams"`: `pos` - An additional mode to get ngrams via PoS tags. It is not mandatory. It only works for `pos`and it is prepared for further development.
    ```jsonlins
      "settings": {
        "tasks": ["features", "features_detail", "plot", "counts", "corpus_characteristics", "compare", "store_sources"],
        "store_sources": true,
        "file_format_features": ["csv"],
        "file_format_clustering": "csv",
        "file_format_plots": ["png", "svg"],
        "file_format_dicts": "txt",
        "boxplot_height": 5,
        "most_frequent_words": 2000,
      },
    ```

## Output: Some Directories

Define different output directories in your configuration file.
Note: The path below the following defined output directories must exist!

* `path_features`
  * for every feature set a subdirectory is created and contains features / metrics in corpus wise tables, format is configurable in settings section, if `counts` has been defined under _tasks_
  * input for _cluster_, _features_detail_
* `path_counts`: 
  * for every feature set a subdirectory is created and contains counts in corpus wise tables, format is configurable in settings section, if `counts` has been defined under _tasks_
* `path_summary`: 
  * contains for every feature set that is calculated on corpus level (macro statistics) a table with results
    * `corpora_characteristics_counts.csv` contains the basic corpus counts of tokens, sentences, documents, etc.
    * corpus wise (macro statistics) measurements: emotion, germanet_semantic_relations, wordnet_semantic_relations, lexical_richness, surface
  * contains for every feature set a subdirectory with a table of every corpus with the summary (e.g. min, max, mean) of features
* `path_features_detail`:
  * summarize single features from feature sets from features
  * if `plot` is set _True_, it contains boxplot visualizations
  * mandatory: run of features before or in same run
* `path_sources`:
  * contains interim results for _compare mode_
  * for every corpus a subdirectory is created with json-files with interim results, e.g. n-gram statistics, synsets of GermaNet and WordNet
  * every file is named by the corpus, feature set, definition of interim result
* `path_clusters`:
  * contains clusters from _cluster mode_ with cluster maps and visualisations
* `path_compare`: 
  * contains results from _compare mode_
  * for every comparing task, a subdirectory is created with visualised plots and tables

    ```jsonlins
      "output": {
        "path_features":        "/path/to/output/features",
        "path_features_detail": "/path/to/output/features_detail",
        "path_summary":         "/path/to/output/summary",
        "path_compare":         "/path/to/output/compare",
        "path_counts":          "/path/to/output/counts",
        "path_sources":         "/path/to/output/sources",
        "path_clusters":        "/path/to/output/clusters"
      }
    ```

**Note: If you run analysis on different devices, copy your output subdirectories and connect the corpus analysis by your interim results and start a new run with a new configuration.
E.g., copy feature results from `dev1/features` and `dev2/features` into `dev/features` and start a new run with `dev/features` as configured features' directory.**

## External Resources

Some features of Semantic Relations need external resources as input:

* [GermaNet](https://uni-tuebingen.de/fakultaeten/philosophische-fakultaet/fachbereiche/neuphilologie/seminar-fuer-sprachwissenschaft/arbeitsbereiche/allg-sprachwissenschaft-computerlinguistik/ressourcen/lexica/germanet-1/)
* Other biomedical dictionaries, look into [semantic_relations.md](semantic_relations.md)
* Self defined dictionaries, examples

    ```jsonlins
    
  "external_resources": {
    "germanet": "/home/chlor/PycharmProjects/feattext/ext_res/germanet/GN_V170/GN_V170_XML",
    "dict":
    {
      "dict_1": "/path/to/dicts_1/",
      "dict_2": "/path/to/dicts_2/"
    }
    ```
## Run

* Open a terminal a type: `python main.py config.json`

## Run tests

* Open a terminal a type: `python -m unittest`

## Logging

* The processing steps are logged, log files stored in the folder `log`.

## Create a package *.exe (Windows)
* Create archive:
  * run `pyinstaller main.py --additional-hooks-dir=hooks-spacy.py`
  * copy the model package `de_core_news_sm` from `venv/lib/python3.10/site-packages/` into `dist/main`
* create an output path, e.g.: `C:\Create\a\path\output`
* Configure [config.json](config.json)
* Run@Windows: open PowerShell or Command Prompt and run `.\\dist\main\main.exe path\\to\\config.json`

----
[Installation](installation.md) | [Input & Data](input.md) | [Functionality & Tasks](tasks.md) | [Feature Hub](features.md) | [Summarization](analytics/summarization.md) | [Comparison](analytics/comparison.md) | [Aggregation](analytics/aggregation.md) | [Config & Run](configuration.md)