import nltk
import numpy as np

from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bow(tokenized_sentence,all_words):
    tokenized_sentence=[stem(w) for w in tokenized_sentence]
    bag=np.zeros(len(all_words),dtype=np.float32) # Initializing with zeros
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1.0 # if word is in tokenized sentence , then its index will become 1
    
    return bag

