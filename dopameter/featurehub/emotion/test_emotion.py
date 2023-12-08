import unittest


class TestLexicalDiversity(unittest.TestCase):

    def test_lexdiv(self):
        text = ('Hallo. Ich bin ein kleiner Blindtext.'
                'Und zwar schon so lange ich denken kann.'
                'Es war nicht leicht zu verstehen, was es bedeutet, ein blinder Text zu sein:'
                'Man ergibt keinen Sinn. Wirklich keinen Sinn.'
                'Man wird zusammenhangslos eingeschoben und rumgedreht – und oftmals gar nicht erst gelesen.'
                'Aber bin ich allein deshalb ein schlechterer Text als andere? Na gut, ich werde nie in den Bestsellerlisten stehen.'
                'Aber andere Texte schaffen das auch nicht.'
                'Und darum stört es mich nicht besonders blind zu sein.'
                'Und sollten Sie diese Zeilen noch immer lesen, so habe ich als kleiner Blindtext etwas geschafft, wovon all die richtigen und wichtigen Texte meist nur träumen.')

        from dopameter.configuration.pipeline import PreProcessingPipline
        nlp = PreProcessingPipline().create_nlp('de')
        doc = nlp(text)

        from dopameter.featurehub.emotion import EmotionFeatures
        emo = EmotionFeatures(lang='de')
        emoemo_metrics = emo.feat_doc(doc=doc)

        self.assertEqual(round(emoemo_metrics['features']['valence'], 2), 5.26)
        self.assertEqual(round(emoemo_metrics['features']['arousal'], 2), 3.87)
        self.assertEqual(round(emoemo_metrics['features']['dominance'], 2), 5.43)
        self.assertEqual(round(emoemo_metrics['features']['joy'], 2), 2.13)

        self.assertEqual(round(emoemo_metrics['features']['anger'], 2), 1.47)
        self.assertEqual(round(emoemo_metrics['features']['sadness'], 2), 1.42)
        self.assertEqual(round(emoemo_metrics['features']['fear'], 2), 1.49)
        self.assertEqual(round(emoemo_metrics['features']['disgust'], 2), 1.48)

        self.assertEqual(round(emoemo_metrics['emotion']['features']['valence'], 2), 179.0)
        self.assertEqual(round(emoemo_metrics['emotion']['features']['arousal'], 2), 131.71)
        self.assertEqual(round(emoemo_metrics['emotion']['features']['dominance'], 2), 184.48)
        self.assertEqual(round(emoemo_metrics['emotion']['features']['joy'], 2), 72.31)

        self.assertEqual(round(emoemo_metrics['emotion']['features']['anger'], 2), 50.12)
        self.assertEqual(round(emoemo_metrics['emotion']['features']['sadness'], 2), 48.45)
        self.assertEqual(round(emoemo_metrics['emotion']['features']['fear'], 2), 50.58)
        self.assertEqual(round(emoemo_metrics['emotion']['features']['disgust'], 2), 50.15)
        self.assertEqual(round(emoemo_metrics['emotion']['sum_emotions_words'], 2), 34)


if __name__ == '__main__':
    unittest.main()
