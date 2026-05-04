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

        from dopameter.featurehub.lexical_diversity import LexicalDiversityFeatures
        lex = LexicalDiversityFeatures()
        lex_metrics = lex.feat_doc(doc=doc)

        self.assertEqual(sum(lex_metrics['lexical_diversity']['freq_list'].values()), 127)
        self.assertEqual(len(lex_metrics['lexical_diversity']['function_words']), 57)
        self.assertEqual(lex_metrics['counts'], {'function_words': 57})

        self.assertEqual(round(lex_metrics['features']['type_token_ratio'], 2), 0.68)
        self.assertEqual(round(lex_metrics['features']['lexical_density'], 2), 44.88)
        self.assertEqual(round(lex_metrics['features']['guiraud_r'], 2), 7.63)
        self.assertEqual(round(lex_metrics['features']['herdan_c'], 2),0.92)
        self.assertEqual(round(lex_metrics['features']['maas_a2'], 2),0.02)
        self.assertEqual(round(lex_metrics['features']['tuldava_ln'], 2),-0.21)
        self.assertEqual(round(lex_metrics['features']['brunet_w'], 2),9.5)
        self.assertEqual(round(lex_metrics['features']['cttr'], 2),5.4)
        self.assertEqual(round(lex_metrics['features']['summer_s'], 2),0.95)
        self.assertEqual(round(lex_metrics['features']['sttr'], 2),0.69)
        self.assertEqual(round(lex_metrics['features']['sichel_s'], 2),0.16)
        self.assertEqual(round(lex_metrics['features']['michea_m'], 2),6.14)
        self.assertEqual(round(lex_metrics['features']['entropy'], 2),4.26)
        self.assertEqual(round(lex_metrics['features']['yule_k'], 2),115.32)
        self.assertEqual(round(lex_metrics['features']['simpson_d'], 2),0.01)
        self.assertEqual(round(lex_metrics['features']['herdan_vm'], 2),0.09)
        self.assertEqual(round(lex_metrics['features']['hdd'], 2),0.84)
        self.assertEqual(round(lex_metrics['features']['evenness'], 2),0.96)
        self.assertEqual(round(lex_metrics['features']['mattr'], 2),0.71)
        self.assertEqual(round(lex_metrics['features']['mtld'], 2),101.05)


if __name__ == '__main__':
    unittest.main()
