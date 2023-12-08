## Installation and Preparation of Language Models and External Resources

* This tool is [Python](https://www.python.org/) based and a Python installation is necessary for usage; preferred version [3.10.6](https://www.python.org/downloads/release/python-3106/) (10/2023).
  * The usage of an environment is preferred, e.g. a _venv_:
  ```commandline
   python -m venv venv
   source venv/bin/activate
  ```

* Install requirements from [requirements.txt](requirements.txt)
 ```requirements.txt
  configparser
  pyinstaller
  numpy
  pandas
  matplotlib
  scikit-learn
  scikit-learn-extra
  spacy
  spacy_syllables
  textacy
  scispacy
  nltk
  benepar
  transformers
  wn
  networkx
  scipy
  openpyxl
  requests
  regex
  setuptools
  seaborn
  protobuf~=3.20.0
  protobuf==4.23.2
```

* Install language models and run `python install_languages.py inst_lang.json` to use resources of _German language_ and _English langauge_:
  * Configure [lang_install.json](../lang_install.json) with the resources that you need.
    ```jsonlines
    {
      "languages": ["de", "en"],
      "modules": ["lang", "wordnet", "const", "emotion"]
    }
    ```
  * `language`:
    * List of languages to install, look into [config_lang.json](../ext_res/installation/config_lang.json) with definition of languages and possible languages for the modules.
    * The definition of the language models is derived from [spaCy models](https://spacy.io/models). The setting of the naming of the language models was updated in August 2023. Please check the naming of the language models before usage.
  * `modules`: List of modules for the installation process
    * `"lang"`: [spaCy language models](https://spacy.io/usage/models)
    * `"wordnet"`: [Open Wordnet](https://pypi.org/project/wn/) / [Open Wordnet @ GitHub](https://github.com/goodmami/wn)
    * `"const"`: [MEmoLon](https://zenodo.org/record/6025889/#.YiW6MhsxnJk) resources for features using emotion scores 
      * Note: [Check](https://askubuntu.com/questions/633176/how-to-know-if-my-gpu-supports-cuda), if your device is CUDA compatible!
      * Note: If you have problems of installation under Ubuntu, try `pip install protobuf==3.20.*`
          * See https://stackoverflow.com/questions/72441758/typeerror-descriptors-cannot-not-be-created-directly
    * `"emotion"`: [Berkeley Parser](https://github.com/nikitakit/self-attentive-parser#available-models) (Constituency Parser)

----
[Installation](installation.md) | [Input & Data](input.md) | [Functionality & Tasks](tasks.md) | [Feature Hub](features.md) | [Summarization](analytics/summarization.md) | [Comparison](analytics/comparison.md) | [Aggregation](analytics/aggregation.md) | [Config & Run](configuration.md)