#coding=utf-8
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from transformers import AutoModelForMaskedLM,RobertaModel
from transformers import AutoModelForMaskedLM
from utils.commonVar import *
from utils.Logger import logger

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pandas as pd
from transformers import logging
logging.set_verbosity_error()
import torch
pd.options.display.max_colwidth = 1000
pd.set_option('display.expand_frame_repr', False)

#classfier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


class IntentBERT(nn.Module):
    def __init__(self, config):
        super(IntentBERT, self).__init__()
        self.device = config['device']
        self.LMName = config['LMName']
        self.clsNum = config['clsNumber']
        self.BN = nn.BatchNorm1d(num_features=768)

        try:
            self.word_embedding = AutoModelForMaskedLM.from_pretrained("./BERT_model")
        except:
            modelPath = os.path.join(SAVE_PATH, self.LMName)
            self.word_embedding = AutoModelForMaskedLM.from_pretrained(os.path.join(SAVE_PATH, self.LMName))
            BNPath = modelPath + '/' + 'BN.pt'
            self.BN.load_state_dict(torch.load(BNPath))

        self.linearClsfier = nn.Linear(768, self.clsNum)
        self.dropout = nn.Dropout(0.1)
        self.word_embedding.to(self.device)
        self.linearClsfier.to(self.device)
        self.BN = self.BN.to(self.device)
        logger.info("Contrastive-learning-based reg: temperature %f."%(config['simTemp']))
        self.sim = Similarity(config['simTemp'])

    def loss_contrastive():
        contrastiveLoss = nn.CrossEntropyLoss()
        output = contrastiveLoss()

    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output
    
    def forward(self, X):
        # BERT forward
        outputs = self.word_embedding(**X, output_hidden_states=True)
        CLSEmbedding_bert = outputs.hidden_states[-1][:,0]

        # linear classifier
        CLSEmbedding = self.dropout(CLSEmbedding_bert)
        logits = self.linearClsfier(CLSEmbedding)

        return CLSEmbedding_bert,logits

    def getUttEmbeddings(self, X, beforeBatchNorm):
        # BERT forward
        outputs = self.word_embedding(**X, output_hidden_states=True)

        # extract [CLS]
        if beforeBatchNorm:
            CLSEmbedding = outputs.hidden_states[-1][:,0]
        else:
            CLSEmbedding = outputs.hidden_states[-1][:,0]
            CLSEmbedding = self.BN(CLSEmbedding)
            CLSEmbedding = self.dropout(CLSEmbedding)


        return CLSEmbedding

    def mlmForward(self, X, Y):
        # BERT forward
        outputs = self.word_embedding(**X, labels=Y)
        return outputs.loss

    # contrastive
    def forwardEmbedding(self, X, beforeBatchNorm=False):
        # get utterances embeddings
        CLSEmbedding = self.getUttEmbeddings(X, beforeBatchNorm = beforeBatchNorm)

        return CLSEmbedding


    def fewShotPredict(self, supportX, supportY, queryX, clsFierName, mode='multi-class'):
        # calculate word embedding
        # BERT forward
        s_embedding = self.word_embedding(**supportX, output_hidden_states=True).hidden_states[-1]
        q_embedding = self.word_embedding(**queryX, output_hidden_states=True).hidden_states[-1]
        
        # extract [CLS] for utterance representation，最后只用了cls的特征
        supportEmbedding = s_embedding[:,0]
        queryEmbedding = q_embedding[:,0]
        support_features = self.normalize(supportEmbedding).cpu()
        query_features = self.normalize(queryEmbedding).cpu()

        # select clsfier
        clf = None
        if clsFierName == CLSFIER_LINEAR_REGRESSION:
            clf = LogisticRegression(penalty='l2',
                                     random_state=0,
                                     C=1.0,
                                     solver='lbfgs',
                                     max_iter=1000,
                                     multi_class='multinomial')
            # fit and predict
            clf.fit(support_features, supportY)
        elif clsFierName == CLSFIER_KNN:
            clf = KNeighborsClassifier()
            clf.fit(support_features, supportY)
        elif clsFierName == CLSFIER_GradientBoostingClassifier:
            clf = GradientBoostingClassifier(n_estimators=200)
            clf.fit(support_features, supportY)
        elif clsFierName == CLSFIER_SVM:
            clf = make_pipeline(StandardScaler(), 
                                SVC(gamma='auto',C=1,
                                kernel='linear',
                                decision_function_shape='ovr'))
            # fit and predict
            clf.fit(support_features, supportY)
        elif clsFierName == CLSFIER_MULTI_LABEL:
            clf = MultiOutputClassifier(LogisticRegression(penalty='l2',
                                                           random_state=0,
                                                           C=1.0,
                                                           solver='liblinear',
                                                           max_iter=1000,
                                                           multi_class='ovr',
                                                           class_weight='balanced'))

            clf.fit(support_features, supportY)
        elif clsFierName == CLSFIER_RONDOM_FOREST:
            clf = RandomForestClassifier(n_estimators=8)
            clf.fit(support_features, supportY)
        elif clsFierName == CLSFIER_AdaBoost:
            clf = AdaBoostClassifier()
            clf.fit(support_features, supportY)


        else:
            raise NotImplementedError("Not supported clasfier name %s", clsFierName)
        
        if mode == 'multi-class':
            query_pred = clf.predict(query_features)#用训练好的分类器去预测
        else:
            logger.error("Invalid model %d"%(mode))

        return query_pred
    
    def reinit_clsfier(self):
        self.linearClsfier.weight.data.normal_(mean=0.0, std=0.02)
        self.linearClsfier.bias.data.zero_()
    
    def set_dropout_layer(self, dropout_rate):
        self.dropout = nn.Dropout(dropout_rate)
    
    def set_linear_layer(self, clsNum):
        self.linearClsfier = nn.Linear(768, clsNum)
    
    def normalize(self, x):
        norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
        out = x.div(norm)
        return out

    def NN(self, support, support_ys, query):
        """nearest classifier"""
        support = np.expand_dims(support.transpose(), 0)
        query = np.expand_dims(query, 2)

        diff = np.multiply(query - support, query - support)
        distance = diff.sum(1)
        min_idx = np.argmin(distance, axis=1)
        pred = [support_ys[idx] for idx in min_idx]
        return pred

    def CosineClsfier(self, support, support_ys, query):
        """Cosine classifier"""
        support_norm = np.linalg.norm(support, axis=1, keepdims=True)
        support = support / support_norm
        query_norm = np.linalg.norm(query, axis=1, keepdims=True)
        query = query / query_norm

        cosine_distance = query @ support.transpose()
        max_idx = np.argmax(cosine_distance, axis=1)
        pred = [support_ys[idx] for idx in max_idx]
        return pred

    def save(self, path):
        self.word_embedding.save_pretrained(path)

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
