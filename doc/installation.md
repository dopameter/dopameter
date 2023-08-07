## Installation and Preparation of Language Models and External Resources

* This tool is [Python](https://www.python.org/) based and necessary for usage; preferred version [3.10.6](https://www.python.org/downloads/release/python-3106/)
  * The usage of a virtual environment is preferred:
    ```commandline
    python -m venv venv
    source venv/bin/activate
    ```

* Install requirements from [requirements.txt](requirements.txt)
  ```requirements.txt
  pyinstaller~=5.11.0
  configparser~=5.3.0
  pandas~=2.0.2
  numpy~=1.24.3
  matplotlib~=3.7.1
  spacy~=3.5.3
  spacy_syllables~=3.0.1
  textacy~=0.13.0
  benepar~=0.2.0
  nltk~=3.7
  wn~=0.9.1
  networkx~=3.1
  scipy~=1.10.1
  openpyxl==3.1.2
  requests~=2.31.0
  scikit-learn~=1.2.2
  scikit-learn-extra
  regex~=2023.5.5
  setuptools~=65.5.1
  transformers~=4.29.2
  seaborn~=0.12.2
  scispacy~=0.5.1
  protobuf~=3.20.0
  ```

  * Install language models and run `python install_languages.py inst_lang.json` to use resources of _German language_ and _English langauge_:
    * configure [lang_install.json](../lang_install.json)
      ```jsonlines
      {
        "languages": ["de", "en"],
        "modules": ["lang", "wordnet", "const", "emotion"]
      }
      ```
    * `language`: list of languages to install, look into [config_lang.json](../ext_res/installation/config_lang.json) with definition of languages and possible languages for the modules 
    * `modules`: list of modules for the installation process
      * `"lang"`: [spaCy language models](https://spacy.io/usage/models)
      * `"wordnet"`: [Open Wordnet](https://pypi.org/project/wn/) / [Open Wordnet @ GitHub](https://github.com/goodmami/wn)
      * `"const"`: [MEmoLon](https://zenodo.org/record/6025889/#.YiW6MhsxnJk) resources for features using emotion scores 
        * Note: [Check](https://askubuntu.com/questions/633176/how-to-know-if-my-gpu-supports-cuda), if your device is CUDA compatible!
        * Note: If you have problems of installation under Ubuntu, try `pip install protobuf==3.20.*`
            * see https://stackoverflow.com/questions/72441758/typeerror-descriptors-cannot-not-be-created-directly
      * `"emotion"`: [Berkeley Parser](https://github.com/nikitakit/self-attentive-parser#available-models) (Constituency Parser)

----
[Installation](../installation.md) | [Input & Data](input.md) | [Functionality & Tasks](tasks.md) | [Feature Hub](features.md) | [Summarization](analytics/summarization.md) | [Comparison](analytics/comparison.md) | [Aggregation](analytics/aggregation.md) | [Config & Run](configuration.md)