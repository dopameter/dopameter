import unittest
import tracemalloc


class TestNER(unittest.TestCase):

    def test_pos(self):
        text = ('Hallo. Ich bin ein kleiner Blindtext. '
                'Und zwar schon so lange ich denken kann. '
                'Es war nicht leicht zu verstehen, was es bedeutet, ein blinder Text zu sein: '
                'Man ergibt keinen Sinn. Wirklich keinen Sinn. '
                'Leipzig, Jena, Chemnitz sind Städte in Mitteldeutschland. '
                'Max Müller und Anton Meier sind Beispielnamen.'
                'Die Friedrich-Schiller-Universität Jena und die Universität Leipzig sind Bildungseinrichtungen.'
                )

        tracemalloc.clear_traces()

        from dopameter.configuration.pipeline import PreProcessingPipline
        nlp = PreProcessingPipline().create_nlp('de')
        doc = nlp(text)

        from dopameter.featurehub.ner import NERFeatures
        ner = NERFeatures(nlp=nlp)
        ner_feat = ner.feat_doc(doc=doc)

        self.assertEqual(ner_feat['counts']['LOC'], 5)
        self.assertEqual(ner_feat['counts']['PER'], 2)
        self.assertEqual(ner_feat['counts']['ORG'], 1)

if __name__ == '__main__':
    unittest.main()
