import unittest


class TestConstituency(unittest.TestCase):

    def test_rb(self):
        text = ('Hallo. Ich bin ein kleiner Blindtext.'
                'Und zwar schon so lange ich denken kann.'
                'Es war nicht leicht zu verstehen, was es bedeutet, ein blinder Text zu sein:'
                'Man ergibt keinen Sinn. Wirklich keinen Sinn.'
                'Man wird zusammenhangslos eingeschoben und rumgedreht – und oftmals gar nicht erst gelesen.'
                'Aber bin ich allein deshalb ein schlechterer Text als andere? Na gut, ich werde nie in den Bestsellerlisten stehen.'
                'Aber andere Texte schaffen das auch nicht.'
                )

        from dopameter.configuration.pipeline import PreProcessingPipline
        nlp = PreProcessingPipline().create_nlp('de')

        from dopameter.features.syntax.constituency import ConstituencyFeatures


        syntax_constituency = ConstituencyFeatures(
            lang='de',
            features={
                'syntax_constituency_metrics': 'default',
                'syntax_constituency_tree': 'default'
        })

        doc = nlp(text)
        sd_feat = syntax_constituency.feat_doc(doc=doc, plain_text=text)

        self.assertEqual( round(sd_feat['syntax_constituency_metrics']['features']['AvgMaxDepth'],2), 4.09)
        self.assertEqual( round(sd_feat['syntax_constituency_metrics']['features']['AvgFan'],2), 2.79)
        self.assertEqual( round(sd_feat['syntax_constituency_metrics']['features']['MaxFan'],2), 6)
        self.assertEqual( round(sd_feat['syntax_constituency_metrics']['features']['AvgNonTerminales_sent'],2), 3.6)
        self.assertEqual( round(sd_feat['syntax_constituency_metrics']['features']['AvgConstituents_sent'],2), 13.0)

        self.assertEqual( round(sd_feat['syntax_constituency_metrics']['features']['AvgTunits_sent'],2), 1.1)
        self.assertEqual( round(sd_feat['syntax_constituency_metrics']['features']['AvgLenConstituents'],2), 2.32)
        self.assertEqual( round(sd_feat['syntax_constituency_metrics']['features']['AvgLenTunits'],2), 5.55)
        self.assertEqual( round(sd_feat['syntax_constituency_metrics']['features']['AvgOutdegreeCentralization'],2), 0.04)
        self.assertEqual( round(sd_feat['syntax_constituency_metrics']['features']['AvgClosenessCentralization'],2), 0.38)


if __name__ == '__main__':
    unittest.main()
