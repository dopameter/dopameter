# DoPA Meter's Updates

## 2026/05/04

* Adjustment of calculations
  * Feature Hub's module _Lexical diversity_:
    * `herdan_c`, `maas_a2`, `sttr`, `michea_m`, `yule_k`, `hdd`, `evenness`, `mattr`
    * changed default value of `window_size=100`
  *  Feature Hub's module _Surface Patterns_:
    * `'ari'`, `'wiener_sachtextformel_1''`, `'wiener_sachtextformel_2''`, `'wiener_sachtextformel_4''`, `'wiener_sachtextformel_2''`

* Improvement in visualizations of aggregation
  * new feature: cluster of k-means can be plotted by t-SNE-plots
  * More configurations of t-SNE-plots via the `"plot"` parameter derived from [seaborn's scatterplot](https://seaborn.pydata.org/generated/seaborn.scatterplot.html)
    * `"alpha": "0.7"` : used for transparency
    * `"marker": "o"` : derived from [matplotlib.markers](https://matplotlib.org/stable/api/markers_api.html)
    * `"s": "1"` : size of markes
    * `"detail_plots" : "false"` : production of detailed plots with visualization of corpora inside clusters 
  * example configuration, for 2 cluster and the default config from sklearn such as the default config of t-SNE:
      ```
          "cluster" : {
            "k-means": {
              "n_clusters": 2,
              "random_state": "None",
              "n_init": "auto",
              "max_iter": 300
            },
            "t-sne": {
              "n_components": 2,
              "random_state": 1,
              "perplexity": 30,
              "learning_rate": 500,
              "init": "pca",
              "plot": {
                "alpha": "0.7",
                "marker": "o",
                "s": "1",
                "detail_plots" : "false"
              }
            },
            "level": ["corpus", "collection"]
          }
      ```
* Improvement in computations of corpus comparisons
  * It is possible to compute comparisons by intersections or differences with the `compare` mode, it is possible to disable the creation of the memory-intensive resource with `    "store_sources": false` as part of the section `settings` (It is the same configuration to store the sources of a corpus by the feature hub.)
  * Update of intersection difference calculation for single vs. all occurrences.

* Some better error reports.
