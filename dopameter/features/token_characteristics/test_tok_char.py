import unittest


class TestTokenCharacteristics(unittest.TestCase):

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

        from dopameter.features.token_characteristics import TokenCharacteristics
        tokchar = TokenCharacteristics()
        tc_feat = tokchar.feat_doc(doc=doc)

        self.assertEqual(tc_feat['counts']['is_alpha'], 109)
        self.assertEqual(tc_feat['counts']['is_ascii'], 124)
        self.assertEqual(tc_feat['counts']['is_digit'], 0)
        self.assertEqual(tc_feat['counts']['is_lower'], 86)
        self.assertEqual(tc_feat['counts']['is_upper'], 0)
        self.assertEqual(tc_feat['counts']['is_title'], 23)
        self.assertEqual(tc_feat['counts']['is_punct'], 18)
        self.assertEqual(tc_feat['counts']['is_left_punct'], 0)
        self.assertEqual(tc_feat['counts']['is_right_punct'], 0)
        self.assertEqual(tc_feat['counts']['is_space'], 0)
        self.assertEqual(tc_feat['counts']['is_bracket'], 0)
        self.assertEqual(tc_feat['counts']['is_quote'], 0)
        self.assertEqual(tc_feat['counts']['is_currency'], 0)
        self.assertEqual(tc_feat['counts']['like_url'], 0)
        self.assertEqual(tc_feat['counts']['like_num'], 0)
        self.assertEqual(tc_feat['counts']['like_email'], 0)
        self.assertEqual(tc_feat['counts']['is_oov'], 127)
        self.assertEqual(tc_feat['counts']['is_stop'], 75)


if __name__ == '__main__':
    unittest.main()
