import unittest


class TestNegation(unittest.TestCase):

    def test_rb(self):
        text = ('Hallo. Ich bin ein kleiner Blindtext, nichts sinnvolles, keine Garantie. '
                'Es wurde kein Verdacht wurde ausgeschlossen. '
                'Es kann nicht in Betracht kommen, dass etwas anders ist. '
                'Es kann nicht in Frage kommen, dass etwas neues kommt. '
                'Man kann man ausschließen, das medizinische Texte keine Negationen beinhalten. '
                'Es gibt keine neue Nachrichten. '
                'Es gibt keine anderen Sorgen, Beschwerden hatte der Patient nicht. '
                'Ein Patient stellt sich in der Klinikambulanz vor. Er berichtet, am Tag zuvor auf der Arbeit Waschbecken auf einen LKW gehoben und dabei plötzlich einen stechenden Schmerz am linken Oberarm verspürt zu haben. Er habe daraufhin nicht mehr weiter arbeiten können und sich daher geschont.'
                'Da die Schmerzen mehr als 12 Stunden später immer noch bestanden, sei er zu seinem Hausarzt gegangen. Dieser habe ihn in die Chirurgische Poliklinik überwiesen. Auf dem Überweisungsschein lesen Sie „V.a.Ruptur der langen Bizepssehne links“.'
                )

        from dopameter.configuration.pipeline import PreProcessingPipline
        nlp = PreProcessingPipline().create_nlp('de')

        from dopameter.features.negation import NegationFeatures
        negation = NegationFeatures(
            nlp=nlp,
            lang='de'
        )

        doc = nlp(text)

        ng_feat=negation.feat_doc(doc=doc)

        self.assertEqual( round(ng_feat['features']['pseudo'],4), 0.0060)
        self.assertEqual( ng_feat['features']['preceding'], 0)
        self.assertEqual( ng_feat['features']['following'], 0)
        self.assertEqual( ng_feat['features']['termination'], 0)
        self.assertEqual( ng_feat['features']['negated_entities'], 0)

        self.assertEqual(ng_feat['counts']['pseudo'], 1)
        self.assertEqual(ng_feat['counts']['preceding'], 0)
        self.assertEqual(ng_feat['counts']['following'], 0)
        self.assertEqual(ng_feat['counts']['termination'], 0)
        self.assertEqual(ng_feat['counts']['negated_entities'], 0)


if __name__ == '__main__':
    unittest.main()
