# Aggregation / Clustering

DOPA METER's aggregation mode is able to compute dependencies between different corpora via 2 modes:
* [_k_-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans),
* [_t_-distributed Stochastic Neighbor Embedding (t-SNE)](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE) with [DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan).

For these steps, [scikit-learn](https://scikit-learn.org) is used.
This modular architecture is open to extend additional clustering algorithms and other machine learning libraries.

# Configuration

* `"corpora""`: Define the corpora for your analysis of clusters!
* `"features"`: Define the feature sets for your analysis of clusters!

* **Decentralized approach:**
  * You can run the task `"cluster"` on different places.
  * You can combine the features via task `"cluster"` if you copy the features `"path_features"` into one directory `"path_features"` and define it into one [configuration file](../resources/example_configurations).    

* Name the 2 modes via in the configuration file, here default is activated:

* `"cluster" : {
    "k-means": "default",
    "t-sne": "default"
  }`

* If you want - add the level data setting of the aggregation mode into the configuration here.
* With the tag `level` you define, if you aggregate (cluster) the data via _corpora_, _collections_ or _languages_.
* It there is not named, all levels _corpora_, _collections_, _languages_ are processed.
* Is there no _collection_ configured in the corpus definition section, _collections_ are _not_ processed.

* `"cluster" : {
    "k-means": "default",
    "t-sne": "default",
    "level": ["corpus", "collection", "language"]
  }`
or
* `"cluster" : {
    "k-means": "default",
    "t-sne": "default",
    "level": ["corpus", "language"]
  }`

## Configuration: _k_-means 

* Note: _k_-means used an internal [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler) of the features before clustering. 
* Default adjustments of _k_-means (following scores loaded automatically)
  * `"n_clusters":` : if you do not define the number of clusters, n is defined by the amount of corpora
  * `"random_state": "None"`
  * `"n_init": 50`
  * `"max_iter": 300`

* Detailed configuration

    ```jsonlines
      "features": {
        "lexical_richness": "default",
        "negation": "default",
        "ner": "default",
        "pos": "default",
        "emotion": "default",
        "token_characteristics": "default",
        "surface": "default",
        "syntax_dependency_metrics" : "default",
        "syntax_dependency_tree": "default"
      },
      "cluster" : {
        "k-means": {
          "n_clusters": 5,
          "random_state": "None",
          "n_init": 50,
          "max_iter": 300
        }
      }
    ```

## Configuration _t_-SNE and DBSCAN

* Default adjustments of t-sne (following scores loaded automatically)
  * `"n_components": 2`
  * `"random_state": 1`
  * `"perplexity": 500` - - Note: be aware, that the defined value of perplexity is at least the amount of single documents.
  * `"learning_rate": 500`
  * `"init": "pca"`
  * `"eps": 2`
  * `"min_samples": 5`

* Detailed configuration

    ```jsonlines
      "features": {
        "lexical_richness": "default",
        "negation": "default",
        "ner": "default",
        "pos": "default",
        "emotion": "default",
        "token_characteristics": "default",
        "surface": "default",
        "syntax_dependency_metrics" : "default",
        "syntax_dependency_tree": "default"
      },
      "cluster" : {
        "t-sne": {
          "n_components": 2,
          "random_state": 1,
          "perplexity": 500,
          "learning_rate": 500,
          "init": "pca",
          "eps": 2,
          "min_samples": 5
        }
      }
    ```

# Output

* For every setup, there will be produced several pictures and cluster maps.
* For every feature, a single directory is produced, including:
  * One figure with clustered data points of the corpora
      ![arc](../resources/example_aggregation/t-sne/token_characteristics/corpus/token_characteristics_corpus_t-sne.png)
  * For every cluster a figure with the highlighted cluster in the map of corpora
      ![arc](../resources/example_aggregation/t-sne/token_characteristics/corpus/token_characteristics_%20corpus%20_t-sne_Cluster_5.png)
  * One figure with clustered data points of the clusters
      ![arc](../resources/example_aggregation/t-sne/token_characteristics/corpus/token_characteristics_t-sne.png)
  * For every corpus a figure with the highlighted cluster in the map of corpora
      ![arc](../resources/example_aggregation/t-sne/token_characteristics/corpus/token_characteristics_tsne_[de.Wiki]%20Wikipedia_ICD-10.png)
  * Cluster maps in table formatted files, [example](../resources/example_aggregation/t-sne/token_characteristics/token_characteristics_tsne_dataset.csv)
  * Summary of cluster maps in 2 Boxplots with the distribution and the occurrences of the amount of documents per cluster
      ![arc](../resources/example_aggregation/t-sne/token_characteristics/frequencies_per_cluster.png)

* [Example files full corpus set-up _k_-means](../resources/example_aggregation/k-means) (feature set _token_characteristics_, features run on 6 collections / 20 corpora)
* [Example files full corpus set-up _t_-SNE](../resources/example_aggregation/t-sne) (feature set _token_characteristics_, features run on 6 collections / 20 corpora)

----
[Installation](../installation.md) | [Input & Data](../input.md) | [Functionality & Tasks](../tasks.md) | [Feature Hub](../features.md) | [Summarization](../analytics/summarization.md) | [Comparison](../analytics/comparison.md) | [Aggregation](../analytics/aggregation.md) | [Config & Run](../configuration.md)