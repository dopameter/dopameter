import unittest


class TestDependency(unittest.TestCase):

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

        from dopameter.features.syntax.dependency import DependencyFeatures


        syntax_dependency = DependencyFeatures(features={
                'syntax_dependency_metrics': 'default',
                'syntax_dependency_tree': 'default'
        })

        doc = nlp(text)
        sd_feat = syntax_dependency.feat_doc(doc=doc)

        for sd in sd_feat['syntax_dependency_metrics']['features']:
            print(sd, round(sd_feat['syntax_dependency_metrics']['features'][sd], 2))

        self.assertEqual( round(sd_feat['syntax_dependency_metrics']['features']['AvgFan'],2), 2.18)
        self.assertEqual( round(sd_feat['syntax_dependency_metrics']['features']['MaxFan'],2), 11)
        self.assertEqual( round(sd_feat['syntax_dependency_metrics']['features']['AvgMaxDepth'],2), 4.59)
        self.assertEqual( round(sd_feat['syntax_dependency_metrics']['features']['AvgDepDist'],2), 2.51)
        self.assertEqual( round(sd_feat['syntax_dependency_metrics']['features']['MaxDepDist'],2), 6.53)
        self.assertEqual( round(sd_feat['syntax_dependency_metrics']['features']['AvgOutdegreeCentralization'],2), 0.46)
        self.assertEqual( round(sd_feat['syntax_dependency_metrics']['features']['AvgClosenessCentralization'],2), 0.37)

        self.assertEqual( sd_feat['syntax_dependency_tree']['counts']['ROOT'], 29)
        self.assertEqual( sd_feat['syntax_dependency_tree']['counts']['punct'], 51)
        self.assertEqual( sd_feat['syntax_dependency_tree']['counts']['nk'], 60)
        self.assertEqual( sd_feat['syntax_dependency_tree']['counts']['svp'], 1)


if __name__ == '__main__':
    unittest.main()
