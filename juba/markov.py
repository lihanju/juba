#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import random
import jieba

class Markov (object):
    def __init__(self, text='', wordLevel= False):
        if wordLevel:
            text = list(text)
        else:
            text = jieba.lcut(text)
        while ' ' in text:
            text.remove(' ')
        self.text = text
        self.length = len(self.text)

    def __randomword__(self,wordList):
        SUM=sum(wordList.values())
        randIndex=random.randint(1,SUM)
        for word,value in wordList.items():
            randIndex -= value
            if randIndex<= 0:
                return word

    def bigram(self):
        wordDict={}
        for i in range(1,self.length):
            if self.text[i - 1] not in wordDict:
                wordDict[self.text[i - 1]]={}
            if self.text[i] not in wordDict[self.text[i - 1]]:
                wordDict[self.text[i - 1]][self.text[i]]=0
            wordDict[self.text[i - 1]][self.text[i]]= wordDict[self.text[i - 1]][self.text[i]] + 1
        return wordDict

    """Generate random documents with a length of textlength starting with firstWord"""
    def random_text(self,textlength,firstWord):
        chain = ""
        currentWord=""
        wordDict=self.bigram()
        if firstWord in wordDict.keys():
            currentWord=firstWord
            for i in range(textlength):
                chain = chain + currentWord
                try:
                    currentWord = self.__randomword__(wordDict[currentWord])
                except:
                    L = [k for k in wordDict.keys()]
                    currentWord = self.__randomword__(wordDict[random.choice(L)])
        else:
            length = len(firstWord)
            for i in range(1, length + 1, 1):
              for key in wordDict.keys():
                if firstWord[-i] in key:
                    currentWord=self.__randomword__(wordDict[key])
                    chain=chain+firstWord
                    for i in range(textlength):
                        chain = chain + currentWord
                        try:
                            currentWord = self.__randomword__(wordDict[currentWord])
                        except:
                            L=[k for k in wordDict.keys()]
                            currentWord = self.__randomword__(wordDict[random.choice(L)])
                    break
              break
            if currentWord=="":
                for key in wordDict.keys():
                    currentWord=key
                    chain=chain+firstWord
                    for i in range(textlength):
                        chain = chain + currentWord
                        try:
                            currentWord = self.__randomword__(wordDict[currentWord])
                        except:
                            L=[k for k in wordDict.keys()]
                            currentWord = self.__randomword__(wordDict[random.choice(L)])
                    break
        chain=chain+"......"
        return chain