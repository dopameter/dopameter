from dopameter.features.token_characteristics import TokenCharacteristics
import spacy
#nlp = spacy.load('de_core_news_sm')

from dopameter.configuration.pipeline import PreProcessingPipline
nlp = PreProcessingPipline().create_nlp('de')

text = 'Das ist ein Text. Und das ist ein Satz. Jetzt kommt etwas mit 3, 4, 5 Beispielen.'
doc = nlp(text)

tok_chars = TokenCharacteristics()
#tc = tok_chars.feat_doc(doc)
#print(tc['counts'])

from dopameter.features.lexical_richness import LexicalRichnessFeatures
lex = LexicalRichnessFeatures()
lex_feat = lex.feat_doc(doc=doc)

print(lex_feat['lexical_richness']['freq_list'])
print(lex_feat['lexical_richness']['corpus_tokens'])
print('function_words', lex_feat['lexical_richness']['function_words'])

print(len(lex_feat['lexical_richness']['freq_list']))
print(len(lex_feat['lexical_richness']['corpus_tokens']))
