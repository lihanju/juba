#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import math

class Similar(object):
    def __init__(self, docs):
        """params:
             self.tf: 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数;
             self.df: 存储每个词及出现了该词的文档数量;
             self.idf: 存储每个词的idf值;
             self.vocabulary: 多个文档docs共有词汇表字典，存储每个词出现总数量;
             self.vocabularyList: 多个文档docs共有词汇列表.
        """
        self.docs = docs
        self.D = len(docs)
        self.avgdl = sum([len(doc)+0.0 for doc in docs]) / self.D
        self.k1 = 1.5
        self.b = 0.75
        self.tf= []
        self.df = {}
        self.idf = {}
        self.vocabulary = {}
        self.vocabularyList=[]
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1
                self.vocabulary[word]=self.vocabulary.get(word, 0) + 1
            self.tf.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D+1)-math.log(v+1)
            self.vocabularyList.append(k)

    #document term matrix(dtm) 文档词汇矩阵
    def tf_dtm(self):
        tf=[]
        length = len(self.vocabularyList)
        for d in range(self.D):
            TF=[]
            for t in range(length):
                if self.vocabularyList[t] not in self.tf[d].keys():
                    TF.append(0)
                else:
                    TF.append(self.tf[d][self.vocabularyList[t]])
            tf.append(TF)
        return tf

    def prob_dtm(self):
        pv=[]
        length = len(self.vocabularyList)
        for d in range(self.D):
            PV=[]
            for t in range(length):
                if self.vocabularyList[t] not in self.tf[d].keys():
                    PV.append(0)
                else:
                    PV.append(self.tf[d][self.vocabularyList[t]]/self.vocabulary[self.vocabularyList[t]])
            pv.append(PV)
        return pv

    def tfidf_dtm(self,norm=False):
        """norm=True,norm='l2'"""
        tfidf=[]
        length = len(self.vocabularyList)
        if norm==True:
            for d in range(self.D):
                TFIDF=[]
                for t in range(length):
                    if self.vocabularyList[t] not in self.tf[d].keys():
                        TFIDF.append(0)
                    else:
                        TFIDF.append(self.tf[d][self.vocabularyList[t]]*self.idf[self.vocabularyList[t]])
                S=math.sqrt(sum([ti*ti for ti in TFIDF]))
                TFIDF=[ti/S for ti in TFIDF]
                tfidf.append(TFIDF)
        if norm == False:
            for d in range(self.D):
                TFIDF = []
                for t in range(length):
                    if self.vocabularyList[t] not in self.tf[d].keys():
                        TFIDF.append(0)
                    else:
                        TFIDF.append(self.tf[d][self.vocabularyList[t]] * self.idf[self.vocabularyList[t]])
                tfidf.append(TFIDF)
        return tfidf

    # term document matrix(tdm) 词汇文档矩阵
    def tf_tdm(self):
        tf={}
        length=len(self.vocabularyList)
        TF=self.tf_dtm()
        for i in range(length):
            tf[self.vocabularyList[i]]=[TF[j][i] for j in range(self.D)]
        return tf

    def prob_tdm(self):
        tf={}
        length=len(self.vocabularyList)
        TF=self.prob_dtm()
        for i in range(length):
            tf[self.vocabularyList[i]]=[TF[j][i] for j in range(self.D)]
        return tf

    def tfidf_tdm(self,norm=False):
        tf={}
        length=len(self.vocabularyList)
        if norm==False:
            TF=self.tfidf_dtm(norm=False)
            for i in range(length):
                tf[self.vocabularyList[i]]=[TF[j][i] for j in range(self.D)]
        if norm == True:
            TF=self.tfidf_dtm(norm=True)
            for i in range(length):
                tf[self.vocabularyList[i]]=[TF[j][i] for j in range(self.D)]
        return tf

    # 计算第一个文档与其他文档的余弦相似度
    def cosine_sim(self, dtm='tfidf_dtm'):
        sim = []
        length = len(self.vocabularyList)
        if dtm== 'tfidf_dtm':
            vec=self.tfidf_dtm()
            for i in range(1,self.D,1):
                    SUM=sum([vec[0][j]*vec[i][j] for j in range(length)])
                    sim.append(SUM)
        if dtm == 'prob_dtm':
            vec = self.prob_dtm()
            S0 = math.sqrt(sum([v * v for v in vec[0]]))
            for i in range(1, self.D, 1):
                SUM = sum([vec[0][j] * vec[i][j] for j in range(length)])
                Si=math.sqrt(sum([v*v for v in vec[i]]))
                sim.append(SUM/(S0*Si))
        if dtm== 'tf_dtm':
            vec = self.tf_dtm()
            S0 = math.sqrt(sum([v * v for v in vec[0]]))
            for i in range(1, self.D, 1):
                SUM = sum([vec[0][j] * vec[i][j] for j in range(length)])
                Si = math.sqrt(sum([v * v for v in vec[i]]))
                sim.append(SUM / (S0 * Si))
        return sim

    # 计算第一个文档与其他文档的权重jaccard相似度
    def weight_jaccard_sim(self):
        S0=sum(self.tf[0].values())
        words=self.tf[0].keys()
        sim=[sum([self.tf[0][w]+self.tf[d][w] for w in words if self.tf[d].get(w,0)>0])/
             (S0+sum(self.tf[d].values())) for d in  range(1,self.D,1)]
        return sim

    # 计算第一个文档与其他文档的jaccard相似度
    def jaccard_sim(self):
        S0=set(self.tf[0].keys())
        sim=[len(S0&set(self.tf[d].keys()))/len(S0|set(self.tf[d].keys())) for d in range(1, self.D, 1)]
        return sim

    def __bm25score__(self,index):
        """index为文档的标记，即第index篇文档"""
        score = 0
        for word in self.docs[0]:
            if word not in self.tf[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word]*self.tf[index][word]*(self.k1+1)
                      / (self.tf[index][word]+self.k1*(1-self.b+self.b*d
                                                      / self.avgdl)))
        return score

    # 计算第一个文档与其他文档的bm25相似度
    def bm25_sim(self):
        sim=[self.__bm25score__(index) for index in range(1,self.D,1)]
        return sim

    # 利用tmd计算两个词汇的相关系数
    def two_term_assocs(self,word_one,word_two,tdm='tf_tdm',norm=False):
        if tdm=='tf_tdm':
            wo=self.tf_tdm()[word_one]
            wt=self.tf_tdm()[word_two]
            wo_mean=sum(wo)/self.D
            wt_mean=sum(wt)/self.D
            assocs=sum([(wo[i]-wo_mean)*(wt[i]-wt_mean) for i in range(self.D)])/\
              (math.sqrt(sum([(wo[i]-wo_mean)**2 for i in range(self.D)]))*math.sqrt(sum([(wt[i]-wt_mean)**2 for i in range(self.D)])))
            return assocs
        if tdm=='prob_tdm':
            wo=self.prob_tdm()[word_one]
            wt=self.prob_tdm()[word_two]
            wo_mean=sum(wo)/self.D
            wt_mean=sum(wt)/self.D
            assocs=sum([(wo[i]-wo_mean)*(wt[i]-wt_mean) for i in range(self.D)])/\
              (math.sqrt(sum([(wo[i]-wo_mean)**2 for i in range(self.D)]))*math.sqrt(sum([(wt[i]-wt_mean)**2 for i in range(self.D)])))
            return assocs
        if tdm=='tfidf_tdm'and norm==False:
            wo=self.tfidf_tdm(norm=False)[word_one]
            wt=self.tfidf_tdm(norm=False)[word_two]
            wo_mean=sum(wo)/self.D
            wt_mean=sum(wt)/self.D
            wo_ss=math.sqrt(sum([(wo[i]-wo_mean)**2 for i in range(self.D)]))
            wt_ss=math.sqrt(sum([(wt[i]-wt_mean)**2 for i in range(self.D)]))
            if wo_ss==0:
                wo_ss=0.000000000000000000001
            if wt_ss==0:
                wt_ss=0.000000000000000000001
            assocs=sum([(wo[i]-wo_mean)*(wt[i]-wt_mean) for i in range(self.D)])/(wo_ss*wt_ss)
            return assocs
        if tdm == 'tfidf_tdm' and norm == True:
            wo = self.tfidf_tdm(norm=True)[word_one]
            wt = self.tfidf_tdm(norm=True)[word_two]
            wo_mean = sum(wo) / self.D
            wt_mean = sum(wt) / self.D
            wo_ss = math.sqrt(sum([(wo[i] - wo_mean) ** 2 for i in range(self.D)]))
            wt_ss = math.sqrt(sum([(wt[i] - wt_mean) ** 2 for i in range(self.D)]))
            if wo_ss == 0:
                wo_ss = 0.000000000000000000001
            if wt_ss == 0:
                wt_ss = 0.000000000000000000001
            assocs = sum([(wo[i] - wo_mean) * (wt[i] - wt_mean) for i in range(self.D)]) / (wo_ss * wt_ss)
            return assocs

    #找出word的相关系数的绝对值不少于mu的所有词汇
    def find_assocs(self,word,mu=0,tdm='tf_tdm',norm=False):
        fa=[]
        length=len(self.vocabularyList)
        if tdm=='tf_tdm':
            for i in range(length):
                tts=self.two_term_assocs(word,self.vocabularyList[i],tdm='tf_tdm')
                if abs(tts)>=mu:
                   fa.append([self.vocabularyList[i],tts])
        if tdm == 'prob_tdm':
            for i in range(length):
                tts=self.two_term_assocs(word,self.vocabularyList[i],tdm='prob_tdm')
                if abs(tts)>=mu:
                   fa.append([self.vocabularyList[i],tts])
        if tdm=='tfidf_tdm'and norm==False:
            for i in range(length):
                tts=self.two_term_assocs(word,self.vocabularyList[i],tdm='tfidf_tdm',norm=False)
                if abs(tts)>=mu:
                   fa.append([self.vocabularyList[i],tts])
        if tdm=='tfidf_tdm'and norm==True:
            for i in range(length):
                tts=self.two_term_assocs(word,self.vocabularyList[i],tdm='tfidf_tdm',norm=True)
                if abs(tts)>=mu:
                   fa.append([self.vocabularyList[i],tts])
        return fa