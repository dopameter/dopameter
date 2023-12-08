import unittest

class TestLexicalDiversity(unittest.TestCase):

    def test_lexdiv(self):
        text = (
            'Hallo. Ich bin ein kleiner Blindtext.'
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

        from dopameter.featurehub.semantics.wordnet import WordNetFeatures
        emo = WordNetFeatures(lang='de')
        sem_metrics = emo.feat_doc(doc=doc)

        print(sem_metrics)

        self.assertEqual(round(sem_metrics['features']['sem_rich'], 2), 18.84)
        self.assertEqual(round(sem_metrics['features']['sem_rich_synonyms'], 2), 5.40)
        self.assertEqual(round(sem_metrics['features']['max_depths'], 2), 2.80)
        self.assertEqual(round(sem_metrics['features']['min_depths'], 2), 2.80)
        self.assertEqual(round(sem_metrics['features']['synsets_avg'], 2), 2.80)
        self.assertEqual(round(sem_metrics['features']['senses_avg'], 2), 2.82)
        self.assertEqual(round(sem_metrics['features']['sem_rich_hyponyms'], 2), 5.46)
        self.assertEqual(round(sem_metrics['features']['sem_rich_taxonyms'], 2), 7.91)
        self.assertEqual(round(sem_metrics['features']['sem_rich_hypernyms'], 2), 2.45)
        self.assertEqual(round(sem_metrics['features']['sem_rich_antonyms'], 2), 5.374)
        self.assertEqual(round(sem_metrics['features']['sem_rich_meronyms'], 2), 0.10)
        self.assertEqual(round(sem_metrics['features']['sem_rich_holonyms'], 2), 0.06)

        self.assertEqual(sem_metrics['doc']['sem_rich'], 2393)
        self.assertEqual(sem_metrics['doc']['synonyms'], 686)
        self.assertEqual(sem_metrics['doc']['max_depths'], 355)
        self.assertEqual(sem_metrics['doc']['min_depths'], 355)
        self.assertEqual(sem_metrics['doc']['synsets_len'], 355)
        self.assertEqual(sem_metrics['doc']['senses_len'], 359)
        self.assertEqual(sem_metrics['doc']['hyponyms'], 693)
        self.assertEqual(sem_metrics['doc']['taxonyms'], 1004)
        self.assertEqual(sem_metrics['doc']['hypernyms'], 311)
        self.assertEqual(sem_metrics['doc']['antonyms'], 682)
        self.assertEqual(sem_metrics['doc']['meronyms'], 13)
        self.assertEqual(sem_metrics['doc']['holonyms'], 8)

if __name__ == '__main__':
    unittest.main()