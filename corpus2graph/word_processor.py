import string
import warnings
import re
from . import util
import spacy
import json

class FileParser(object):
    def __init__(self,
                 file_parser='txt',
                 xml_node_path=None,
                 json_attribute=None, fparser=None):
        if file_parser not in ['txt', 'xml', 'json', 'defined']:
            msg = 'file_parser should be txt, xml or defined, not "{file_parser}"'
            raise ValueError(msg.format(file_parser=file_parser))
        if file_parser == 'defined' and fparser is None:
            msg = 'Please define you own file_parser.'
            raise ValueError(msg)
        self.file_parser = file_parser
        self.xml_node_path = xml_node_path
        self.json_attribute = json_attribute
        self.fparser = fparser

    def xml_parser(self, file_path, xml_node_path):
        for paragraph in util.search_all_specific_nodes_in_xml_known_node_path(file_path, xml_node_path):
            for sent in util.tokenize_informal_paragraph_into_sentences(paragraph):
                yield str.strip(sent)

    def txt_parser(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield str.strip(line)

    def json_parser(self, file_path, json_attribute):
        with open(file_path, 'r', encoding='utf-8') as file:
            #TODO: Support json attribute path
            for sent in util.tokenize_informal_paragraph_into_sentences(json.load(file)[json_attribute]):
                yield str.strip(sent)

    

    def __call__(self, file_path):
        if self.file_parser == 'txt':
            for sent in self.txt_parser(file_path):
                yield sent
        if self.file_parser == 'xml':
            for sent in self.xml_parser(file_path, self.xml_node_path):
                yield sent
        if self.file_parser == 'json':
            for sent in self.json_parser(file_path, self.json_attribute):
                yield sent
        if self.file_parser == 'defined':
            for sent in self.fparser(file_path):
                yield sent


class WordPreprocessor(object):
    # default: config file.
    def __init__(self, remove_stop_words, remove_numbers, replace_digits_to_zeros, remove_punctuations,
                 stem_word, lowercase, wpreprocessor):
        self.remove_stop_words = remove_stop_words
        self.remove_numbers = remove_numbers
        self.replace_digits_to_zeros = replace_digits_to_zeros
        self.remove_punctuations = remove_punctuations
        self.stem_word = stem_word
        self.lowercase = lowercase
        self.wpreprocessor = wpreprocessor

        punctuations = set(string.punctuation)
        punctuations.update({'“', '”', '—'})  # English
        punctuations.update({'...', '«', '»'})  # French
        self.puncs = punctuations

    def apply(self, word, spacy_loader=None):
        # Removing
        if self.remove_numbers and word.isnumeric():
            return ''
        if self.replace_digits_to_zeros:
            word = re.sub('\d', '0', word)
        if self.remove_punctuations:
            if all(c in self.puncs for c in word):
                return ''
        # Remove combinations of punctuations and digits
        if self.remove_numbers and self.remove_punctuations:
            if all(j.isdigit() or j in self.puncs for j in word):
                return ''
        # remove stop words
        if self.remove_stop_words and spacy_loader.vocab[word].is_stop:
            # print(word, 'is stop words')
            return ''

        # Stem word
        if self.stem_word:
            word = util.stem_word(word)

        # Make all words in lowercase
        if self.lowercase:
            word = word.lower()

        # customized word preprocessor
        if self.wpreprocessor is not None:
            if not callable(self.wpreprocessor):
                msg = 'wpreprocessor should be callable'
                warnings.warn(msg)
            else:
                word = self.wpreprocessor(word)
                if not isinstance(word, str):
                    msg = 'The output of wpreprocessor should be string'
                    raise ValueError(msg)
        return word

    def __call__(self, word):
        return self.apply(word)


class Tokenizer(object):
    @staticmethod
    def mytok(s):
        """
        An example of user customized tokenizer.
        :return: list of tokens
        """
        # TODO NOW spacy.load here is a really stupid idea, cause each time apply has been called spacy.load need to run. TOO SLOW!!!
        tk = spacy.load('en')
        return [token.text for token in tk(s)]

    def __init__(self, word_tokenizer='Treebank', wtokenizer=None):
        self.word_tokenizer = None

        if word_tokenizer not in ['Treebank', 'PunktWord', 'WordPunct', 'spacy', '']:
            msg = 'word_tokenizer "{word_tokenizer}" should be Treebank, PunktWord, WordPunct or empty'
            raise ValueError(msg.format(word_tokenizer=word_tokenizer))
        if word_tokenizer == 'spacy':
            self.tokenizer = None
            self.word_tokenizer = 'spacy'
        elif word_tokenizer == 'Treebank':
            from nltk.tokenize import TreebankWordTokenizer
            self.tokenizer = TreebankWordTokenizer().tokenize
        elif word_tokenizer == 'PunktWord':
            # PunktTokenizer splits on punctuation, but keeps it with the word. => [‘this’, “‘s”, ‘a’, ‘test’]
            from nltk.tokenize import PunktWordTokenizer
            self.tokenizer = PunktWordTokenizer().tokenize
        elif word_tokenizer == 'WordPunct':
            # WordPunctTokenizer splits all punctuations into separate tokens. => [‘This’, “‘”, ‘s’, ‘a’, ‘test’]
            from nltk.tokenize import WordPunctTokenizer
            self.tokenizer = WordPunctTokenizer().tokenize
        else:
            if wtokenizer is None:
                self.tokenizer = None
            else:
                if not callable(wtokenizer):
                    msg = 'wtokenizer should be callable'
                    warnings.warn(msg)
                    self.tokenizer = None
                else:
                    self.tokenizer = wtokenizer

    def apply(self, text, spacy_loader=None):
        if self.word_tokenizer == 'spacy':
            return [token.text for token in spacy_loader(text)]
        if self.tokenizer is not None:
            return self.tokenizer(text)
        else:
            return [text]

    def __call__(self, text):
        return self.apply(text)
