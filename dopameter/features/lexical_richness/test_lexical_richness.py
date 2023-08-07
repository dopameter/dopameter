import unittest


class TestLexicalRichness(unittest.TestCase):

    def test_lexrich(self):
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

        from dopameter.features.lexical_richness import LexicalRichnessFeatures
        lex = LexicalRichnessFeatures()
        lex_feat = lex.feat_doc(doc=doc)

        self.assertEqual(sum(lex_feat['lexical_richness']['freq_list'].values()), 127)
        self.assertEqual(len(lex_feat['lexical_richness']['function_words']), 57)
        self.assertEqual(lex_feat['counts'], {'function_words': 57})

        self.assertEqual(round(lex_feat['features']['type_token_ratio'], 2), 0.68)
        self.assertEqual(round(lex_feat['features']['lexical_density'], 2), 44.88)
        self.assertEqual(round(lex_feat['features']['guiraud_r'], 2), 7.63)
        self.assertEqual(round(lex_feat['features']['herdan_c'], 2),2.82)
        self.assertEqual(round(lex_feat['features']['maas_a2'], 2),58.44)
        self.assertEqual(round(lex_feat['features']['tuldava_ln'], 2),-0.21)
        self.assertEqual(round(lex_feat['features']['brunet_w'], 2),9.5)
        self.assertEqual(round(lex_feat['features']['cttr'], 2),5.4)
        self.assertEqual(round(lex_feat['features']['summer_s'], 2),0.95)
        self.assertEqual(round(lex_feat['features']['sttr'], 2),1.2)
        self.assertEqual(round(lex_feat['features']['sichel_s'], 2),0.16)
        self.assertEqual(round(lex_feat['features']['michea_m'], 2),1893.64)
        self.assertEqual(round(lex_feat['features']['entropy'], 2),5.71)
        self.assertEqual(round(lex_feat['features']['yule_k'], 2),3.07)
        self.assertEqual(round(lex_feat['features']['simpson_d'], 2),0.03)
        self.assertEqual(round(lex_feat['features']['herdan_vm'], 2),0.09)
        self.assertEqual(round(lex_feat['features']['hdd'], 2),0.1)
        self.assertEqual(round(lex_feat['features']['evenness'], 2),3.19)
        self.assertEqual(round(lex_feat['features']['mattr'], 2),0.91)
        self.assertEqual(round(lex_feat['features']['mtld'], 2),101.05)


if __name__ == '__main__':
    unittest.main()
