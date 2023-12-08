import unittest


class TestPOS(unittest.TestCase):

    def test_pos(self):
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

        from dopameter.featurehub.pos import POSFeatures
        pos = POSFeatures(nlp=nlp)
        pos_metrics = pos.feat_doc(doc=doc)

        self.assertEqual(pos_metrics['counts']['PROPN'], 1)
        self.assertEqual(pos_metrics['counts']['PUNCT'], 18)
        self.assertEqual(pos_metrics['counts']['PRON'], 16)
        self.assertEqual(pos_metrics['counts']['AUX'], 10)
        self.assertEqual(pos_metrics['counts']['DET'], 9)
        self.assertEqual(pos_metrics['counts']['ADJ'], 7)
        self.assertEqual(pos_metrics['counts']['NOUN'], 10)
        self.assertEqual(pos_metrics['counts']['CCONJ'], 8)
        self.assertEqual(pos_metrics['counts']['ADV'], 22)
        self.assertEqual(pos_metrics['counts']['VERB'], 14)
        self.assertEqual(pos_metrics['counts']['PART'], 7)
        self.assertEqual(pos_metrics['counts']['ADP'], 3)
        self.assertEqual(pos_metrics['counts']['INTJ'], 1)
        self.assertEqual(pos_metrics['counts']['SCONJ'], 1)

if __name__ == '__main__':
    unittest.main()
