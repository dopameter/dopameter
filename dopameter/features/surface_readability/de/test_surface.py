import unittest


class TestSurface(unittest.TestCase):

    def test_rb(self):
        text = ('Hallo. Ich bin ein kleiner Blindtext.'
                'Und zwar schon so lange ich denken kann.'
                'Es war nicht leicht zu verstehen, was es bedeutet, ein blinder Text zu sein:'
                'Man ergibt keinen Sinn. Wirklich keinen Sinn.'
                'Man wird zusammenhangslos eingeschoben und rumgedreht – und oftmals gar nicht erst gelesen.'
                'Aber bin ich allein deshalb ein schlechterer Text als andere? Na gut, ich werde nie in den Bestsellerlisten stehen.'
                'Aber andere Texte schaffen das auch nicht.'
                'Und darum stört es mich nicht besonders blind zu sein.'
                'Und sollten Sie diese Zeilen noch immer lesen, so habe ich als kleiner Blindtext etwas geschafft, wovon all die richtigen und wichtigen Texte meist nur träumen.'
                ' Dies ist ein Typoblindtext. An ihm kann man sehen, ob alle Buchstaben da sind und wie sie aussehen.'
                'Manchmal benutzt man Worte wie Hamburgefonts, Rafgenduks oder Handgloves, um Schriften zu testen.'
                'Manchmal Sätze, die alle Buchstaben des Alphabets enthalten – man nennt diese Sätze »Pangrams«.'
                'Sehr bekannt ist dieser: The quick brown fox jumps over the lazy old dog.'
                'Oft werden in Typoblindtexte auch fremdsprachige Satzteile eingebaut (AVAIL® and Wefox™ are testing aussi la Kerning), um die Wirkung in anderen Sprachen zu testen.'
                'In Lateinisch sieht zum Beispiel fast jede Schrift gut aus. Quod erat demonstrandum. Seit 1975 fehlen in den meisten Testtexten die Zahlen, weswegen nach TypoGb. 204 § ab dem Jahr 2034 Zahlen in 86 der Texte zur Pflicht werden.'
                'Nichteinhaltung wird mit bis zu 245 € oder 368 $ bestraft.'
                'Genauso wichtig in sind mittlerweile auch Âçcèñtë, die in neueren Schriften aber fast immer enthalten sind. Ein wichtiges aber schwierig zu integrierendes Feld sind OpenType-Funktionalitäten.'
                'Je nach Software und Voreinstellungen können eingebaute Kapitälchen, Kerning oder Ligaturen (sehr pfiffig) nicht richtig dargestellt werden. Dies ist ein Typoblindtext.'
                'An ihm kann man sehen, ob alle Buchstaben da sind und wie sie aussehen. Manchmal benutzt man Worte wie Hamburgefonts, Rafgenduks …'
                )

        from dopameter.configuration.pipeline import PreProcessingPipline
        nlp = PreProcessingPipline().create_nlp('de')

        import spacy_syllables  # Do not delete this line!
        nlp.add_pipe("syllables", after="tagger")
        from dopameter.features.surface_readability.de import SurfaceFeaturizesDE

        doc = nlp(text)
        surface = SurfaceFeaturizesDE()
        rb_feat = surface.feat_doc(doc=doc)

        self.assertEqual(rb_feat['counts']['toks_min_three_syllables'], 48)
        self.assertEqual(rb_feat['counts']['toks_larger_six_letters'], 84)
        self.assertEqual(rb_feat['counts']['toks_one_syllable'], 169)
        self.assertEqual(rb_feat['counts']['syllables'], 500)
        self.assertEqual(rb_feat['counts']['letter_tokens'], 302)
        self.assertEqual(rb_feat['counts']['no_digit_tokens'], 308)

        self.assertEqual( round(rb_feat['features']['avg_token_len_chars'],2), 4.64)
        self.assertEqual( round(rb_feat['features']['avg_sent_len_tokens'],2), 12.41)
        self.assertEqual( round(rb_feat['features']['avg_sent_len_chars'],2), 67.28)
        self.assertEqual( round(rb_feat['features']['flesch_kincaid_grade_level'],2), 7.71)
        self.assertEqual( round(rb_feat['features']['smog'],2),  0)
        self.assertEqual( round(rb_feat['features']['coleman_liau'],2), 11.94)
        self.assertEqual( round(rb_feat['features']['ari'],2), 8.62)
        self.assertEqual( round(rb_feat['features']['forcast'],2), 14.37)
        self.assertEqual( round(rb_feat['features']['gunning_fog'],2),  10.48)
        self.assertEqual( round(rb_feat['features']['wiener_sachtextformel_1'],2), 1.25)
        self.assertEqual( round(rb_feat['features']['wiener_sachtextformel_2'],2),  -0.63)
        self.assertEqual( round(rb_feat['features']['wiener_sachtextformel_3'],2), 1.29)
        self.assertEqual( round(rb_feat['features']['wiener_sachtextformel_4'],2),1.17)
        self.assertEqual( round(rb_feat['features']['flesch_reading_ease'],2), 84.42)
        self.assertEqual( round(rb_feat['features']['heylighen_formality'],2), 59.85)

if __name__ == '__main__':
    unittest.main()
