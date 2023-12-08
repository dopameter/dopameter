import os
import collections
import csv
import logging


class EmotionFeatures:
    """Get metrics of emotion

    Parameters
    ----------
    lang : string
    features : list, [
            'valence',
            'arousal',
            'dominance',
            'joy',
            'anger',
            'sadness',
            'fear',
            'disgust'
        ]

    Attributes
    ----------
    * `data` : dictionary with emotion scores as keys:
        * `valence` : average score of valence scores and words with emotions
        * `arousal` : average score of arousal scores and words with emotions
        * `dominance` : average score of dominance scores and words with emotions
        * `joy` : average score of valence joy and words with emotions
        * `anger` : average score of anger scores and words with emotions
        * `sadness` : average score of sadness scores and words with emotions
        * `fear` : average score of fear scores and words with emotions
        * `disgust` : average score of disgust scores and words with emotions

    Notes
    -----
    define in configuration .json file under features:

    * default:
      "tasks": ["features"],
      "features": {
        "surface": "default"
      }
    * or in detail:
      "tasks": ["features"],
      "features": {
        "surface": [
          "valence",
          "arousal",
          "dominance",
          "joy",
          "anger",
          "sadness",
          "fear",
          "disgust"
          ]
        }


    """

    def __init__(
            self,
            lang,
            features='default'
    ):

        logging.info('\tInitialize emotion features.')

        self.lang = lang

        self.emotion_lang_path = os.path.join(
            os.sep.join(os.path.abspath(__file__).split(os.sep)[:-4]),
            'ext_res',
            'emotion',
            lang + '.tsv'
        )

        default_features = [
            'valence',
            'arousal',
            'dominance',
            'joy',
            'anger',
            'sadness',
            'fear',
            'disgust'
        ]

        if features == 'default':
            self.features = default_features
        else:
            if set(features).intersection(default_features) == set():
                raise ValueError('Your emotion features ' + ' '.join(self.features) + ' are not defined! Allowed definitions: ', default_features)
            else:
                logging.info('\t\tDefined features: ' + str(features))
                self.features = features

        self.emotion_scores = {}
        with open(self.emotion_lang_path, 'r', encoding='utf-8') as data_file:
            data = csv.DictReader(data_file, delimiter="\t")
            for row in data:
                word = row['word']
                self.emotion_scores[word] = row['word']
                row.pop('word')
                self.emotion_scores[word] = row
        self.emotion_scores_keys = self.emotion_scores.keys()

        logging.info('\t\tMEmoLon emotion scores loaded from ' + self.emotion_lang_path)


    def feat_doc(self, doc):
        """Compute metrics of surface patterns for a document

        Parameters
        ----------
        doc : spaCy Doc

        Returns
        -------
        dict
            dictionary with 'features' as key including the metrics and 'surface' with interim results and 'counts' including amount of countable metrics
        """

        words = dict(collections.Counter([tok.text for tok in doc if (not tok.is_punct) and (not tok.is_stop)]))
        emotions = {feat: 0 for feat in self.features}
        sum_words = 0

        data = {
            'features': {},
            'emotion':
                {
                    'features': {feat: 0 for feat in self.features},
                    'sum_emotions_words': {}
                }
        }

        for word in words:
            if word in self.emotion_scores_keys:
                sum_words = sum_words + words[word]
                for feat in self.features:
                    score = float(words[word] * float(self.emotion_scores[word][feat]))
                    emotions[feat] = emotions[feat] + score
                    data['emotion']['features'][feat] = data['emotion']['features'][feat] + score

        data['emotion']['sum_emotions_words'] = sum_words

        if sum_words > 0:
            data['features'] = {emotion: (emotions[emotion] / sum_words) for emotion in emotions}
        else:
            data['features'] = {emotion: 0 for emotion in emotions}

        return data

    def feat_corpus(self, corpus):

        """Featurize a corpus with emotion  metrics

        Parameters
        ----------

        Returns
        -------
        dict
            dictionary with 'features' as key including the corpus metrics
        """

        if corpus.resources.emotion['sum_emotions_words'] > 0:
            return {'features':{em: (corpus.resources.emotion['features'][em] / corpus.resources.emotion['sum_emotions_words']) for em in corpus.resources.emotion['features']}}
        else:
            return {'features': {em: 0 for em in corpus.resources.emotion['features']}}

def init_emotion(features):
    return {
        'sum_emotions_words': 0,
        'features': {feat: 0 for feat in features}
    }

def update_emotion(emotion, data):
    if 'sum_emotions_words' in data.keys():
        emotion['sum_emotions_words'] += data['sum_emotions_words']
    emotion['features'] = dict(collections.Counter(emotion['features']) + collections.Counter(data['features']))

    return emotion