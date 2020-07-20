# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:33:43 2020

@author: yanik
"""

import csv
import numpy as np

#store sentences and tags into a map (1)
def prepareData(dataFile):
    
    DATA = { "Sentence#":[] , "Words":[], "Tag": [] }
    
    sentenceCount = 0
    with open(dataFile,'r', encoding="utf8") as f:
        reader=csv.reader(f,delimiter='\t')
        for element in reader:
            if (len(element) != 0):
                DATA["Sentence#"].append(sentenceCount)
                DATA["Words"].append(element[0])
                DATA["Tag"].append(element[1])
            else:
                sentenceCount = sentenceCount + 1
                
    return DATA


#returns all sentences from dataset (3)
def getSentence(DATA):
    
    sentenceNum = DATA["Sentence#"]
    words = DATA["Words"]
    tags = DATA["Tag"]
    
    aWord = []
    allSentences = []
    
    aWordTag = []
    
    wordIndex = 0
    currrentCount = 0
    
    for count in sentenceNum:
        
        if (count != currrentCount):
            mapped = list(zip(aWord, aWordTag))
            aWord = []
            aWordTag = []
            allSentences.append(mapped)
            currrentCount = count
            
        aWord.append(words[wordIndex])
        aWordTag.append(tags[wordIndex])
        
        wordIndex = wordIndex + 1
            
                    
    return allSentences

#returns words and tags and the amounts (2)
def get_words_and_tags(DATA): 
    
    words = list(set(DATA["Words"]))
    #since length of sentences can vary, add a special tag at the end of them
    words.append("ENDPAD")
    tags = list(set(DATA["Tag"]))
    
    n_words = len(words)
    n_tags = len(tags)

    return words, tags, n_words, n_tags


#check how long the senctences are. (can be called before 4)
def plot_sentences_long_graph(sentences, bins=20, style="ggplot"):
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")
    
    plt.hist([len(s) for s in sentences], bins=bins, align='left')
    plt.title('Length of sentences')
    plt.xlabel('word count')
    plt.ylabel('Amount of sentences')
    plt.xticks(range(26))
    plt.show()
    

#Construct word and tag dictionaries (4)
def word_tag_dictionaries(words, tags, max_len=20):
    word2idx = {w: i for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    
    return word2idx, tag2idx

