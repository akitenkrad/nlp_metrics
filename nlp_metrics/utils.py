'''utils'''
import numpy as np
import nltk
from nltk import word_tokenize
from collections import defaultdict

class Coder(object):
    '''provide encoder and decoder between sentence and indices'''
 
    UNKNOWN = '<UNK>'
    
    def __init__(self, case_sensitive=False):
        self.encoder = defaultdict(lambda: 0)
        self.decoder = defaultdict(lambda: self.UNKNOWN)
        self.case_sensitive = case_sensitive
 
    @property
    def tokenizer(self):
        return nltk.word_tokenize

    @property   
    def vocab(self):
        return self.encoder

    def build(self, dataset:list):
        '''build a dictionary
        
        Args:
            dataset: a list of sentence
        '''
        self.encoder[self.UNKNOWN] = 0
        self.decoder[0] = self.UNKNOWN
        
        for sentence in dataset:
            words = self.tokenizer(sentence)
            for word in words:
                _word = word.lower() if self.case_sensitive == False else word
                if _word not in self.encoder:
                    self.encoder[_word] = len(self.encoder)
                    self.decoder[self.encoder[_word]] = _word

    def word2idx(self, word:str):
        return self.encoder[word]
    
    def idx2word(self, idx:int):
        return self.decoder[idx]
    
    def encode(self, sentence):
        words = sentence
        if isinstance(sentence, str):
            words = self.tokenizer(sentence)
            words = [w.lower() for w in words] if self.case_sensitive == False else words
        return [self.word2idx(word) for word in words]
    
    def decode(self, indices:list):
        return [self.idx2word(idx) for idx in indices]
    