from dopameter.featurehub.token_characteristics import TokenCharacteristics
import spacy

from dopameter.configuration.pipeline import PreProcessingPipline
nlp = PreProcessingPipline().create_nlp('de')

text = 'Das ist ein Text. Und das ist ein Satz. Jetzt kommt etwas mit 3, 4, 5 Beispielen.'
doc = nlp(text)

tok_chars = TokenCharacteristics()

from dopameter.featurehub.lexical_diversity import LexicalDiversityFeatures
lex = LexicalDiversityFeatures()
lex_feat = lex.feat_doc(doc=doc)

print(lex_feat['lexical_diversity']['freq_list'])
print(lex_feat['lexical_diversity']['corpus_tokens'])
print('function_words', lex_feat['lexical_diversity']['function_words'])

print(len(lex_feat['lexical_diversity']['freq_list']))
print(len(lex_feat['lexical_diversity']['corpus_tokens']))
