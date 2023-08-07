# -*- coding: utf-8 -*-
"""
The delta.corpus module contains code for building, loading, saving, and
manipulating the representation of a corpus. Its heart is the :class:`Corpus`
class which represents the feature matrix. Also contained are default
implementations for reading and tokenizing files and creating a feature vector
out of that.
out of that.
"""
import json
import os
import glob
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatch
from inspect import signature
from typing import Optional, Union

import regex as re
import pandas as pd
import collections
import csv
from math import ceil
from dopameter.delta.util import Metadata, DocumentDescriber, DefaultDocumentDescriber, ngrams

from joblib import Parallel, delayed

import logging

LETTERS_PATTERN = re.compile(r'\p{L}+')
WORD_PATTERN = re.compile(r"\b(\p{L}[\p{L}'’]*?|[\p{L}'’]*?\p{L})\b", re.WORD)


@dataclass(frozen=True)
class FeatureGenerator:
    """
    A **feature generator** is responsible for converting a subdirectory of
    files into a feature matrix (that will then become a corpus). If you need
    to customize the feature extraction process, create a custom feature
    generator and pass it into your :class:`Corpus` constructor call along with
    its `subdir` argument.

    The default feature generator is able to process a directory of text files,
    tokenize each of the text files according to a regular expression, and
    count each token type for each file. To customize feature extraction, you
    have two options:

        1. for simple customizations, just create a new FeatureGenerator and
           set the constructor arguments accordingly. Look in the docstring for
           :meth:`__init__` for details.
        2. in more complex cases, create a subclass and override methods as you
           see fit.

    On a feature generator passed in to :class:`Corpus`, only two methods will
    be called:

        * :meth:`__call__`, i.e. the object as a callable, to actually generate
            the feature vector,
        * :attr:`metadata` to obtain metadata fields that will be included in
            the corresponding corpus.

    So, if you wish to write a completely new feature generator, you can ignore
    the other methods.


    Args:
        lower_case (bool): if ``True``, normalize all tokens to lower case
            before counting them
        encoding (str): the encoding to use when reading files
        glob (str): the pattern inside the subdirectory to find files.
        skip (str): don't handle files that match this pattern
        token_pattern (re.Regex): The regular expression used to identify
            tokens. The default, LETTERS_PATTERN, will simply find sequences
            of unicode letters. WORD_PATTERN will find the shortest sequence
            of letters and apostrophes between two word boundaries
            (according to the simple word-boundary algorithm from *Unicode
            regular expressions*) that contains at least one letter.
        max_tokens (int): If set, stop reading each file after that many words.
        ngrams (int): Count token ngrams instead of single tokens
        parallel(bool, int, Parallel): If truish, read and parse files in parallel. The actual argument may be
            - None or False for no special processing
            - an int for the required number of jobs
            - a dictionary with Parallel arguments for finer control
        sort (str): Sort the final feature matrix by index before returning. Possible values:
            - ``documents``, ``index``: Sort by document names
            - ``features``, ``columns``: sort by feature labels (ie words)
            - ``both``: sort along both axes
            - None or the empty string: Do not sort
        sparse (bool): build a sparse dataframe. Requires Pandas >=1.0
    """

    lower_case: bool = False
    encoding: str = "utf-8"
    glob: str = "*.txt"
    skip: Optional[str] = None
    token_pattern: re.Pattern = LETTERS_PATTERN
    max_tokens: Optional[int] = None
    ngrams: Optional[int] = None
    parallel: Union[int, bool, Parallel] = False
    sort: str = 'documents'
    sparse: bool = False
    logger = logging.getLogger(__name__ + '.FeatureGenerator')

    def tokenize(self, lines):
        """
        Tokenizes the given lines.

        This method is called by :meth:`count_tokens`. The default
        implementation will return an iterable of all tokens in the given
        :param:`lines` that matches the :attr:`token_pattern`. The result
        of this method can further be postprocessed by
        :meth:`postprocess_tokens`.

        Args:
            lines: Iterable of strings in which to look for tokens.

        Returns:
            Iterable (default implementation generator) of tokens
        """
        count = 0
        for line in lines:
            for match in self.token_pattern.finditer(line):
                count += 1
                yield match.group(0)
                if self.max_tokens is not None and count >= self.max_tokens:
                    return

    def postprocess_tokens(self, tokens):
        """
        Postprocesses the tokens according to the options provided when
        creating the feature generator..

        Currently respects `lower_case` and `ngrams`. This is called by
        count_tokens after tokenizing.

        Args:
            tokens: iterable of tokens as returned by :meth:`tokenize`

        Returns:
            iterable of postprocessed tokens
        """
        if self.lower_case:
            tokens = (token.lower() for token in tokens)

        if self.ngrams:
            tokens = ngrams(tokens, n=self.ngrams, sep=" ")

        return tokens

    def count_tokens(self, lines):
        """
        This calls :meth:`tokenize` to split the iterable `lines` into tokens.
        If the :attr:`lower_case` attribute is given, the tokens are then
        converted to lower_case. The tokens are counted, the method returns a
        :class:`pd.Series` mapping each token to its number of occurrences.

        This is called by :meth:`process_file`.

        Args:
            lines: Iterable of strings in which to look for tokens.
        Returns:
            pandas.Series: maps tokens to the number of occurrences.
        """
        tokens = self.postprocess_tokens(self.tokenize(lines))
        count = collections.defaultdict(int)
        for token in tokens:
            count[token] += 1
        return pd.Series(count)

    def get_name(self, filename):
        """
        Converts a single file name to a label for the corresponding feature
        vector.

        Returns:
            str: Feature vector label (filename w/o extension by default)
        """
        return os.path.basename(filename).rsplit('.', 1)[0]

    def process_file(self, filename):
        """
        Processes a single file to a feature vector.

        The default implementation reads the file pointed to by `filename` as a
        text file, calls :meth:`count_tokens` to create token counts and
        :meth:`get_name` to calculate the label for the feature vector.

        Args:
            filename (str): The path to the file to process
        Returns:
            :class:`pd.Series`: Feature counts, its name set according to
                :meth:`get_name`
        """
        self.logger.info("Reading %s ...", filename)
        with open(filename, "rt", encoding=self.encoding) as file:
            series = self.count_tokens(file)
            if series.name is None:
                series.name = self.get_name(filename)
            return series

    def process_directory(self, directory):
        """
        Iterates through the given directory and runs :meth:`process_file` for
        each file matching :attr:`glob` in there.

        Args:
            directory (str): Path to the directory to process

        Returns:
            dict: mapping name to :class:`pd:Series`
        """
        filenames = glob.glob(os.path.join(directory, self.glob))
        if len(filenames) == 0:
            self.logger.error(
                    "No files matching %s in %s. Feature matrix will be empty.",
                    self.glob,
                    directory)
        else:
            self.logger.info(
                    "Reading %d files matching %s from %s",
                    len(filenames),
                    self.glob,
                    directory)
        used_filenames = [filename for filename in filenames if self.skip is None or not (fnmatch(filename, self.skip))]
        parallel = self._get_parallel_executor()
        if parallel:
            data = parallel(delayed(self.process_file)(filename) for filename in used_filenames)
        else:
            data = (self.process_file(filename) for filename in used_filenames)
        return {self.get_name(fn): series for (series, fn) in zip(data, used_filenames)}

    def _get_parallel_executor(self) -> Parallel:
        if self.parallel:
            if isinstance(self.parallel, Mapping):
                return Parallel(**self.parallel)
            elif self.parallel == True:
                return Parallel(-1)
            elif isinstance(self.parallel, int):
                return Parallel(self.parallel)
        return None

    def __call__(self, directory): # orig code line
    #def __call__(self, list_of_ngram_files):
        """
        Runs the feature extraction using :meth:`process_directory` for the
        given directory and returns a simple pd.DataFrame for that. The resulting
        dataframe will be sorted according to the `sort` attribute.
        """

        #print(list_of_ngram_files)
        #corpora = {}
        #for file in list_of_ngram_files:
        #    with open(list_of_ngram_files[file], encoding='utf-8') as json_file:
        #        corpora[file] = json.load(json_file)
        #df = pd.DataFrame(corpora).transpose().fillna(0)
        #df.index.name = 'corpus'
        #print(df)
        #return df

        data = self.process_directory(directory)
        if self.sparse:
            dtype = pd.SparseDtype(pd.Int64Dtype, pd.NA)
        else:
            dtype = pd.Int64Dtype

        # df = pd.DataFrame.from_dict(data, orient='index', dtype=dtype) # orig. code line
        df = pd.DataFrame(data).transpose()

        if self.sort:
            if self.sort.lower() in {'documents', 'index', 'both'}:
                df = df.sort_index(axis=0)
            if self.sort.lower() in {'features', 'columns', 'both'}:
                df = df.sort_index(axis=1)
        return df

    @property
    def metadata(self):
        """
        Returns:
            Metadata: metadata record that describes the parameters of the
                features used for corpora created using this feature generator.
        """
        return Metadata(features='words', lower_case=self.lower_case)


#class _NamedCounter(collections.Counter):
#
#    def __init__(self, iterable=None, _name='', **kwds):
#        super().__init__(iterable, **kwds)
#        self.name = _name


class SimpleFeatureGenerator(FeatureGenerator):
    """
    A simplified, faster version of the FeatureGenerator.

    With respect to feature generation the behaviour is the same as with FeatureGenerator, but it is slightly less
    flexible with respect to subclassing. It does not read the files linewise, and it never creates pd.Series().
    """

    def preprocess_text(self, text):
        if self.lower_case:
            return text.lower()
        else:
            return text

    def postprocess_tokens(self, tokens):
        if self.ngrams:
            tokens = ngrams(tokens, n=self.ngrams, sep=" ")
        return tokens

    def process_file(self, filename):
        """ adapted by Christina Lohr """

        with open(filename, encoding=self.encoding) as f:
            text = self.preprocess_text(f.read())
            tokens = self.postprocess_tokens(self.tokenize([text]))

            return dict(collections.Counter(tokens))

            #return _NamedCounter(tokens, self.get_name(filename))


class CorpusNotComplete(ValueError):
    def __init__(self, msg="Corpus not complete anymore"):
        super().__init__(msg)


class CorpusNotAbsolute(CorpusNotComplete):
    def __init__(self, operation):
        super().__init__("{} not possible: Absolute frequencies required.".format(operation))


class Corpus(pd.DataFrame):
    _metadata = ['metadata', 'logger', 'document_describer', 'feature_generator']

    def __init__(
            self,
            source=None,
            *,
            subdir=None,
            file=None,
            corpus=None,
            feature_generator=None,
            document_describer=DefaultDocumentDescriber(),
            metadata=None, **kwargs):
        """
        Creates a new Corpus.

        You can create a corpus either from a filesystem subdir with raw text files, or from a CSV file with
        a document-term matrix, or from another corpus or dataframe that contains (potentially preprocessed)
        document/term vectors. Either option may be passed via appropriately named keyword argument or as
        the only positional argument, but exactly one must be present.

        If you pass a subdirectory, Corpus will call a `FeatureGenerator` to read and parse the files and to
        generate a default word count. The default implementation will search for plain text files ``*.txt``
        inside the directory and parse them using a simple regular expression. It has a few options, e.g.,
        ``glob`` and ``lower_case``, that can also be passed directly to corpus as keyword arguments. E.g.,
        ``Corpus('texts', glob='plain-*.txt', lower_case=True)`` will look for files called plain-xxx.txt and
        convert it to lower case before tokenizing. See `FeatureGenerator` for more details.

        The ``document_describer`` can contain per-document metadata which can be used, e.g, as ground truth.

        The ``metadata`` record contains global metadata (e.g., which transformations have already been performed),
        they will be inherited from a ``corpus`` argument, all additional keyword arguments will be included
        with this record.

        Args:
            source: Positional variant of either subdir, file, or corpus

        Keyword Args:
            subdir (str): Path to a subdirectory containing the (unprocessed) corpus data.
            file (str): Path to a CSV file containing the feature vectors.
            corpus (pandas.DataFrame): A dataframe or :class:`Corpus` from which to create a new corpus, as a copy.
            feature_generator (FeatureGenerator): A customizeable helper class that will process a `subdir` to a feature matrix, if the `subdir` argument is also given. If None, a default feature generator will be used.
            metadata (dict): A dictionary with metadata to copy into the new corpus.
            **kwargs: Additionally, if feature_generator is None and subdir is not None, you can pass FeatureGenerator
                arguments and they will be used when instantiating the feature generator
                Additional keyword arguments will be set in the metadata record of the new corpus.

        Warning:
            You should either use a single positional argument (source) or one of subdir, file, or corpus as
            keyword arguments.  In future versions, source will be positional-only.
        """
        logger = logging.getLogger(__name__)

        # Check and normalize the parameters so we have either file or subdir or corpus != None
        if sum(1 for arg in [source, subdir, file, corpus] if arg is not None) != 1:
            raise ValueError('Exactly one of the positional argument, subdir, file or corpus must be present')
        if source is not None:
            if isinstance(source, str) or isinstance(source, os.PathLike):
                if os.path.isfile(source):
                    file = source
                else:
                    subdir = source
            else:
                corpus = source

        # initialize or update metadata
        if metadata is None:
            metadata = Metadata(
                    ordered=False,
                    words=None,
                    corpus=subdir if subdir else file,
                    complete=True,
                    frequencies=False)
        else:
            metadata = Metadata(metadata)  # copy it, just in case

        # initialize data
        if subdir is not None:
            if feature_generator is None:  # generate default feature generator from matching args
                fg_sig_arguments = signature(SimpleFeatureGenerator).parameters
                fg_actual_args = {}
                for key, value in kwargs.copy().items():
                    if key in fg_sig_arguments:
                        fg_actual_args[key] = value
                        del kwargs[key]  # if they belong in metadata, FeatureGenerator will put them there
                feature_generator = SimpleFeatureGenerator(**fg_actual_args)

            logger.info(
                    "Creating corpus by reading %s using %s",
                    subdir,
                    feature_generator)
            df = feature_generator(subdir)
            metadata.update(feature_generator)
        elif file is not None:
            logger.info("Loading corpus from CSV file %s ...", file)
            df = pd.read_csv(file, index_col=0).T
            try:
                metadata = Metadata.load(file)
            except OSError:
                logger.warning(
                        "Failed to load metadata for %s. Using defaults: %s",
                        file,
                        metadata,
                        exc_info=False)

        elif corpus is not None:
            df = corpus
            if isinstance(corpus, Corpus):
                metadata.update(corpus.metadata)
            elif not isinstance(corpus, pd.DataFrame):

                print(df)

                df = pd.DataFrame(df)
        else:
            assert False, 'we already checked above that one of corpus, file, or subdir is not None'

        metadata.update(**kwargs)

        if not metadata.ordered:
            df = df.iloc[:, (-df.sum()).argsort()]
            metadata.ordered = True

        super().__init__(df.fillna(0))
        self.logger = logger
        self.metadata = metadata
        self.document_describer = document_describer
        self.feature_generator = feature_generator

    def new_data(self, data, **metadata):
        """
        Wraps the given `DataFrame` with metadata from this corpus object.

        Args:
            data (pandas.DataFrame): Raw data that is derived by, e.g., pandas filter operations
            **metadata: Metadata fields that should be changed / modified
        """
        return Corpus(corpus=data,
                      feature_generator=self.feature_generator,
                      document_describer=self.document_describer,
                      metadata=Metadata(self.metadata, **metadata))

    def save(self, filename="corpus_words.csv"):
        """
        Saves the corpus to a CSV file.

        The corpus will be saved to a CSV file containing documents in the
        columns and features in the rows, i.e. a transposed representation.
        Document and feature labels will be saved to the first row or column,
        respectively.

        A metadata file will be saved alongside the file.


        Args:
            filename (str): The target file.
        """
        self.logger.info("Saving corpus to %s ...", filename)
        self.T.to_csv(
                filename,
                encoding="utf-8",
                na_rep=0,
                quoting=csv.QUOTE_NONNUMERIC)
        self.metadata.save(filename)

    def is_absolute(self) -> bool:
        """
        Returns:
            bool: ``True`` if this is a corpus using absolute frequencies
        """
        return not (self.metadata.frequencies)

    def is_complete(self) -> bool:
        """
        A corpus is complete as long as it contains the absolute frequencies of
        all features of all documents. Many operations like calculating the
        relative frequencies require a complete corpus. Once a corpus has lost
        its completeness, it is not possible to restore it.
        """
        return self.metadata.complete

    def get_mfw_table(self, mfwords):
        """
        Shortens the list to the given number of most frequent words and converts
        the word counts to relative frequencies.

        This returns a new :class:`Corpus`, the data in this object is not modified.

        Args:
            mfwords (int): number of most frequent words in the new corpus. 0 means all words.

        See also:
            :meth:`Corpus.top_n`, :meth:`Corpus.relative_frequencies`

        Returns:
            Corpus: a new sorted corpus shortened to `mfwords`
        """
        if mfwords > 0:
            return self.relative_frequencies().top_n(mfwords)
        else:
            return self.relative_frequencies()

    def top_n(self, mfwords):
        """
        Returns a new `Corpus` that contains the top n features.

        Args:
            mfwords (int): Number of most frequent items in the new corpus.

        Returns:
            Corpus: a new corpus shortened to `mfwords`
        """
        return Corpus(
                corpus=self.iloc[:, :mfwords],
                document_describer=self.document_describer,
                metadata=self.metadata,
                complete=False,
                words=mfwords)

    def save_wordlist(self, filename, **kwargs):
        """
        Saves the current word list to a text file.

        Args:
            filename (str): Path to the file to save
            kwargs: Additional arguments to pass to :func:`open`
        """
        with open(filename, 'w', **kwargs) as out:
            out.write("# One word per line. Empty lines or "
                      "lines starting with # are ignored.\n\n")
            for word in self.columns:
                out.write(word + '\n')

    def _load_wordlist(self, filename, **kwargs):
        """
        Loads the given word list.

        Args:
            filename (str): Name of file to load
            kwargs: Additional arguments for `open`

        Yields:
            Features from the given file
        """
        ENTRY = re.compile(r'\s*([^#]+)')
        with open(filename, 'r', **kwargs) as f:
            for line in f:
                match = ENTRY.match(line)
                if match:
                    feature = match.group(1).strip()
                    if feature:
                        yield feature

    def filter_wordlist(self, filename, **kwargs):
        """
        Returns a new corpus that contains the features from the given file.

        This method will read the list of words from the given file and then
        return a new corpus that uses the features listed in the file, in the
        order they are in the file.

        Args:
            filename (str):
                Path to the file to load. Each line contains one feature.
                Leading and trailing whitespace, lines starting with ``#``, and
                empty lines are ignored.

        Returns:
            New corpus with seelected features.
        """
        words = list(self._load_wordlist(filename, **kwargs))
        return self.filter_features(words, wordlist=filename)

    def filter_features(self, features, **metadata):
        """
        Returns a new corpus that contains only the given features.

        Args:
            features (Iterable):
                The features to select. If its in a file, use filter_wordlist
        """
        return self.new_data(self.loc[:, features],
                             complete=False,
                             **metadata)

    def relative_frequencies(self):
        if self.metadata.frequencies:
            return self
        elif not (self.is_complete()):
            raise CorpusNotComplete()
        else:
            new_corpus = self.div(self.sum(axis=1), axis=0)
            return Corpus(
                    corpus=new_corpus,
                    document_describer=self.document_describer,
                    metadata=self.metadata,
                    complete=False,
                    frequencies=True)

    def z_scores(self):
        df = (self - self.mean()) / self.std()
        return Corpus(corpus=df,
                      document_describer=self.document_describer,
                      metadata=self.metadata,
                      z_scores=True,
                      complete=False,
                      frequencies=True)

    def cull(self, ratio=None, threshold=None, keepna=False):
        """
        Removes all features that do not appear in a minimum number of
        documents.

        Args:
            ratio (float): Minimum ratio of documents a word must occur in to
                be retained. Note that we're always rounding towards the
                ceiling, i.e.  if the corpus contains 10 documents and
                ratio=1/3, a word must occur in at least *4* documents (if this
                is >= 1, it is interpreted as threshold)
            threshold (int): Minimum number of documents a word must occur in
                to be retained
            keepna (bool): If set to True, the missing words in the returned
                corpus will be retained as ``nan`` instead of ``0``.

        Returns:
            Corpus: A new corpus witht the culled words removed. The original
                corpus is left unchanged.
        """
        if ratio is not None:
            if ratio > 1:
                threshold = ratio
            else:
                threshold = ceil(ratio * self.index.size)
        elif threshold is None:
            return self

        culled = self.replace(0, float('NaN')).dropna(thresh=threshold, axis=1)
        if not keepna:
            culled = culled.fillna(0)
        return Corpus(corpus=culled, complete=False,
                      document_describer=self.document_describer,
                      metadata=self.metadata, culling=threshold)

    def reparse(self, feature_generator, subdir=None, **kwargs):
        """
        Parse or re-parse a set of documents with different settings.

        This runs the given feature generator on the given or configured
        subdirectory. The feature vectors returned by the feature generator
        will replace or augment the corpus.

        Args:
            feature_generator (FeatureGenerator): Will be used for extracting
                stuff.
            subdir (str): If given, will be passed to the feature generator for
                processing. Otherwise, we'll use the subdir configured with
                this corpus.
            **kwargs: Additional metadata for the returned corpus.
        Returns:
            Corpus: a new corpus with the respective columns replaced or added.
                The current object will be left unchanged.
        Raises:
            CorpusNotAbsolute: if called on a corpus with relative frequencies
        """
        if not (self.is_absolute()):
            raise CorpusNotAbsolute('Replacing or adding documents')
        if subdir is None:
            if self.metadata.corpus is not None \
                    and os.path.isdir(self.metadata.corpus):
                subdir = self.metadata.corpus
        reparsed = feature_generator(subdir)
        df = pd.DataFrame(self, copy=True)
        for new_doc in reparsed.index:
            df.loc[new_doc, :] = reparsed.loc[new_doc, :]
        return Corpus(corpus=df, metadata=self.metadata, **kwargs)

    def tokens(self) -> pd.Series:
        """Number tokens by text"""
        if self.is_absolute():
            return self.sum(axis=1)
        else:
            raise CorpusNotAbsolute('Calculation on absolute numbers')

    def types(self) -> pd.Series:
        """Number of different features by text"""
        if self.is_absolute():
            return self.replace(0, float('NaN')).count(axis=1)
        else:
            raise CorpusNotAbsolute('Calculation on absolute numbers')

    def ttr(self) -> float:
        """
        Type/token ratio for the whole corpus.

        See also:
            https://en.wikipedia.org/wiki/Lexical_density
        """
        return self.types().sum() / self.tokens().sum()

    def ttr_by_text(self) -> pd.Series:
        """
        Type/token ratio for each text.
        """
        return self.types() / self.tokens()
