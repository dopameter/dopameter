import unittest

class TestSurface(unittest.TestCase):

    def test_rb(self):
        text = (
            "In publishing and graphic design, Lorem ipsum is a placeholder text commonly used to demonstrate the visual"
            " form of a document or a typeface without relying on meaningful content. Lorem ipsum may be used as a "
            "placeholder before final copy is available. It is also used to temporarily replace text in a process called"
            " greeking, which allows designers to consider the form of a webpage or publication, without the meaning of"
            " the text influencing the design. Lorem ipsum is typically a corrupted version of De finibus bonorum et"
            " malorum, a 1st-century BC text by the Roman statesman and philosopher Cicero, with words altered, added,"
            " and removed to make it nonsensical and improper Latin. The first two words themselves are a truncation of"
            " dolorem ipsum ('pain itself'). Versions of the Lorem ipsum text have been used in typesetting at least "
            "since the 1960s, when it was popularized by advertisements for Letraset transfer sheets. Lorem ipsum "
            "was introduced to the digital world in the mid-1980s, when Aldus employed it in graphic and word-processing"
            " templates for its desktop publishing program PageMaker. Other popular word processors, including Pages and"
            " Microsoft Word, have since adopted Lorem ipsum,[2] as have many LaTeX packages, web content managers such"
            " as Joomla! and WordPress, and CSS libraries such as Semantic UI."
        )

        from dopameter.configuration.pipeline import PreProcessingPipline
        nlp = PreProcessingPipline(config = {'features' : {'surface': 'default'}}).create_nlp('en')

        import spacy_syllables  # Do not delete this line!
        nlp.add_pipe("syllables", after="tagger")
        from dopameter.featurehub.surface.en import SurfaceFeaturizesEN

        doc = nlp(text)
        surface = SurfaceFeaturizesEN()
        surface_metrics = surface.feat_doc(doc=doc)

        self.assertEqual(surface_metrics['counts']['toks_min_three_syllables'], 32)
        self.assertEqual(surface_metrics['counts']['toks_larger_six_letters'], 69)
        self.assertEqual(surface_metrics['counts']['toks_one_syllable'], 125)
        self.assertEqual(surface_metrics['counts']['syllables'], 334)
        self.assertEqual(surface_metrics['counts']['letter_tokens'], 214)
        self.assertEqual(surface_metrics['counts']['no_digit_tokens'], 214)

        self.assertEqual( round(surface_metrics['features']['avg_token_len_chars'],2), 4.59)
        self.assertEqual( round(surface_metrics['features']['avg_sent_len_tokens'],2), 24.3)
        self.assertEqual( round(surface_metrics['features']['avg_sent_len_chars'],2), 131.7)
        self.assertEqual( round(surface_metrics['features']['flesch_kincaid_grade_level'],2), 11.17)
        self.assertEqual( round(surface_metrics['features']['smog'],2),  0)
        self.assertEqual( round(surface_metrics['features']['coleman_liau'],2), 12.66)
        self.assertEqual( round(surface_metrics['features']['ari'],2), 13.17)
        self.assertEqual( round(surface_metrics['features']['forcast'],2), 13.75)
        self.assertEqual( round(surface_metrics['features']['gunning_fog'],2),  14.54)
        self.assertEqual( round(surface_metrics['features']['flesch_reading_ease'],2), 53.07)
        self.assertEqual( round(surface_metrics['features']['heylighen_formality'],2), 79.27)
        self.assertEqual( round(surface_metrics['features']['dale_chall'],2), 11.93)

if __name__ == '__main__':
    unittest.main()
