# Surface Patterns

* Also known as _Readability_ scores, mainly focus on syllables, token and sentence length criteria and thus target surface-level indicators for linguistic complexity only.

We also provide the number of tokens with at least three syllables, tokens larger six letters and tokens consisting one
syllable as configurable features of the feature set, which is the input of the four WSF.
The amount of syllables can be used with the corpus comparison mode, also the mentioned three WSF input scores.

* Interim results / Counts - configurable, but not part of defaults:
  * *toks_min_three_syllables*: list / amount tokens with a minimum of syllables
  * *toks_larger_six_letters*: list / amount of tokens larger six letters
  * *toks_one_syllable*: list / amount of tokens with one syllable
  * *syllables*: single syllables
  * *letter_tokens*: words without punctuations and without digits
  * *no_digit_tokens*: words without punctuations

* Metrics:
  * *avg_token_len_chars*: average token length measured by count of characters
  * *avg_sent_len_tokens*: average sentences length measured by count of tokens
  * *avg_sent_len_chars*: average sentences measured by count of characters
    
  * *dale_chall*: Dale–Chall readability formula measured by list of 3000 words, works only for English language (Dale, Chall, 1995)
    $[0.1579 ((\text{difficult words} / \text{words}) * 100) + 0.0496 (\text{words} / \text{sentences})]$
  * *flesch_reading_ease*: Flesch-Reading Ease score adapted for English language and German language (Flesch, 1948)
      * English (Rudolf Flesch 1948):
        $[206.835 - 1.015 (\text{count of tokens} / \text{count of sentences}) - 84.6 (\text{count of  syllables} / \text{count of words})]$
      * German (Toni Armstad 1978):
        $[180 - (\text{count of tokens} / \text{count of  sentences}) - 58.5 (\text{count of  syllables} / \text{count of  words})]$
    * *flesch_kincaid_grade_level*: Flesch–Kincaid grade level (Flesch, 1943):
        $[0.39 * (\text{total words} / \text{total sentences}) + 11.8 (\text{total syllables} / \text{total words}) - 15.59]$
    * *smog*: Simple Measure of Gobbledygook (SMOG) grade (McLaughlin, 1969) - only with document level
       $[1,0430 sqrt{ \text{count of polysyllables} * (30 / \text{count of sentences} ) } + 3.1291]$
    * *coleman_liau*: Coleman–Liau index (Coleman, Liau 1975)
        $[0.0588 * (\text{average number of letters per 100 words} - 0.296 * (\text{average number of sentences per 100 words}) - 15.8 ]$
    * *ari*: automated readability index (Senter, Smith, 1967)
        $[4.71 (\text{characters} / \text{tokens}) + 0.5 (\text{tokens} / \text{sentences}) - 21.43 ]$
    * *forcast*: FORCAST formula (US military, 1973) - only with document level
        $[20 - ((\text{number of single-syllable words in a 150 word sample}) / 10)]$
    * *gunning_fog*: Gunning fog index (Gunning, 1952)
        $[0.4 * ((\text{words} / \text{sentences}) + 100 * (\text{complex words} / \text{word}))]$ 
    * *Wiener Sachtext formula: only German language*
      * *wiener_sachtextformel_1*: First Wiener Sachtextformel (Richard Bamberger and Erich Vanecek, 1984)
         $[(0.1935 * \text{words with 3 or more syllables}) + (0.16772 * \text{average sentence length}) + (0.1297 * \text{words with more than 6 letters}) - (0.0327 * es) - 0.875]$
      * *wiener_sachtextformel_2*: Second Wiener Sachtextformel (Richard Bamberger and Erich Vanecek, 1984)
         $[(0.2007 * \text{words with 3 or more syllables}) + (0.1682 * \text{average sentence length}) + (0.1373 * \text{words with more than 6 letters}) - 2.779 ]$
      * *wiener_sachtextformel_3*: Third Wiener Sachtextformel (Richard Bamberger and Erich Vanecek, 1984)
         $[ (0.2963 * \text{words with 3 or more syllables}) + (0.1905 * \text{average sentence length}) - 1.1144]$
      * *wiener_sachtextformel_4*: Fourth Wiener Sachtextformel (Richard Bamberger and Erich Vanecek, 1984)
         $[ 0.2744 * (\text{words with 3 or more syllables} + 0.2656) * (\text{average sentence length} - 1.693)]$ 
    * *heylighen_formality* : F-Score (formality score as defined by Heylighen and Dewaele (1999)
         $[(\text{noun freq} + \text{adjective freq} + \text{preposition freq} + \text{article freq} - \text{pronoun freq} - \text{verb freq} - \text{adverb freq} - \text{interjection freq} + 100) / 2]$

* Note:
  * The package is inspired by the spaCy extension 'Readability', what is implemented for an older spacy version (< 3) and English Language:
      * https://spacy.io/universe/project/spacy_readability
      * https://github.com/mholtzscher/spacy_readability/tree/master/spacy_readability

* Configuration with defaults:

```jsonlines
  "tasks": ["features", "counts"],
  "features": {
    "surface": "default"
  }
```
* Configuration in detail:

```jsonlines
   "tasks": ["features", "counts"],
  "features": {
    "surface": [
        "toks_min_three_syllables",
        "toks_larger_six_letters",
        "toks_one_syllable",
        "syllables",
        "letter_tokens",
        "no_digit_tokens",
        "avg_token_len_chars",
        "avg_sent_len_tokens",
        "avg_sent_len_chars",
        "flesch_kincaid_grade_level",
        "smog",
        "coleman_liau",
        "ari",
        "forcast",
        "gunning_fog",
        "heylighen_formality"
        ]
  }
```

----
[Basic Counts (Token and Corpus Characteristics, POS, NER)](./basics.md) | [N-Grams](./ngrams.md) | [Lexical Diversity](./lexical_diversity.md) | [Surface Patterns](./surface.md) | [Syntax](./syntax.md) | [Semantic Relations](./semantic_relations.md) | [Emotion](features/emotion.md)

----
[Installation](../installation.md) | [Input & Data](../input.md) | [Functionality & Tasks](../tasks.md) | [Feature Hub](../features.md) | [Summarization](../analytics/summarization.md) | [Comparison](../analytics/comparison.md) | [Aggregation](../analytics/aggregation.md) | [Config & Run](../configuration.md)