from utils.IntentDataset import IntentDataset
from utils.Evaluator import EvaluatorBase
from utils.Logger import logger
from utils.commonVar import *
from utils.tools import mask_tokens, makeTrainExamples
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
from sklearn import metrics
from sklearn.metrics import accuracy_score, r2_score
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
##
# @brief  base class of trainer
class TrainerBase():
    def __init__(self):
        self.finished=False
        self.bestModelStateDict = None
        self.roundN = 4
        self.eps = 1e-6
        pass

    def round(self, floatNum):
        return round(floatNum, self.roundN)

    def train(self):
        raise NotImplementedError("train() is not implemented.")

    def getBestModelStateDict(self):
        return self.bestModelStateDict

class TransferTrainer(TrainerBase):
    def __init__(self,
                   trainingParam:dict,
                 optimizer,
                 dataset:IntentDataset,
                 unlabeled:IntentDataset,
                 conhe: IntentDataset,
                 valEvaluator: EvaluatorBase,
                 testEvaluator:EvaluatorBase):

        super(TransferTrainer, self).__init__()
        self.epoch       = trainingParam['epoch']
        self.batch_size  = trainingParam['batch']
        self.validation  = trainingParam['validation']
        self.patience    = trainingParam['patience']
        self.tensorboard = trainingParam['tensorboard']
        self.mlm         = trainingParam['mlm']
        self.lambda_mlm  = trainingParam['lambda mlm']
        self.regression  = trainingParam['regression']
        self.lossContrastiveWeight = trainingParam['lossContrastiveWeight']

        self.dataset       = dataset
        self.unlabeled     = unlabeled
        self.conhe           = conhe
        self.optimizer     = optimizer
        self.valEvaluator  = valEvaluator
        self.testEvaluator = testEvaluator
        self.temperature = 0.05
        if self.tensorboard:
            self.writer = SummaryWriter()

        self.batchMonitor = trainingParam["batchMonitor"]

        self.beforeBatchNorm = trainingParam['beforeBatchNorm']
        logger.info("In trainer, beforeBatchNorm %s"%(self.beforeBatchNorm))

    def duplicateInput(self, X):
        batchSize = X['input_ids'].shape[0]

        X_duplicate = {}
        X_duplicate['input_ids'] = X['input_ids'].unsqueeze(1).repeat(1,2,1).view(batchSize*2, -1)
        X_duplicate['token_type_ids'] = X['token_type_ids'].unsqueeze(1).repeat(1,2,1).view(batchSize*2, -1)
        X_duplicate['attention_mask'] = X['attention_mask'].unsqueeze(1).repeat(1,2,1).view(batchSize*2, -1)

        return X_duplicate
    def calculateDropoutCLLoss(self, model, X, beforeBatchNorm=False):
        # duplicate input
        batch_size = X['input_ids'].shape[0]
        X_dup = self.duplicateInput(X)

        # get raw embeddings
        batchEmbedding = model.forwardEmbedding(X_dup, beforeBatchNorm=beforeBatchNorm)
        batchEmbedding = batchEmbedding.view((batch_size, 2, batchEmbedding.shape[1])) # (bs, num_sent, hidden)

        # Separate representation
        z1, z2 = batchEmbedding[:,0], batchEmbedding[:,1]

        cos_sim = model.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        logits = cos_sim

        labels = torch.arange(logits.size(0)).long().to(model.device)
        lossVal = model.loss_ce(logits, labels)

        return  lossVal

    def ContrastiveLoss(self, model, X, X_1,X_2, conloader):
        anchor, positive, negative = self.posneg(model, X, X_1, X_2, conloader,beforeBatchNorm=False)

        # 计算相似性
        sim_pos = torch.cosine_similarity(anchor, positive) / self.temperature
        sim_neg = torch.cosine_similarity(anchor, negative) / self.temperature

        # 使用对比损失
        loss = torch.mean(torch.logsumexp(sim_neg, dim=0) - sim_pos)
        return loss

    def train(self, model, tokenizer, mode='multi-class'):
        self.bestModelStateDict = copy.deepcopy(model.state_dict())
        durationOverallTrain = 0.0
        durationOverallVal = 0.0
        valBestAcc = -1
        accumulateStep = 0

        # evaluate before training
        valAcc, valPre, valRec, valFsc = self.valEvaluator.evaluate(model, tokenizer, mode)
        teAcc, tePre, teRec, teFsc = self.testEvaluator.evaluate(model, tokenizer, mode)
        logger.info('---- Before training ----')
        logger.info("ValAcc %f, Val pre %f, Val rec %f , Val Fsc %f", valAcc, valPre, valRec, valFsc)
        logger.info("TestAcc %f, Test pre %f, Test rec %f, Test Fsc %f", teAcc, tePre, teRec, teFsc)

        if mode == 'multi-class':
            labTensorData = makeTrainExamples(self.dataset.getTokList(), tokenizer, self.dataset.getLabID(), mode=mode)
        else:
            logger.error("Invalid model %d"%(mode))
        dataloader = DataLoader(labTensorData, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        
        if self.mlm:
            unlabTensorData = makeTrainExamples(self.unlabeled.getTokList(), tokenizer, mode='unlabel')
            unlabeledloader = DataLoader(unlabTensorData, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=False)
            unlabelediter = iter(unlabeledloader)

        if self.conhe:
            conheTensorData = makeTrainExamples(self.conhe.getTokList(), tokenizer, mode='unlabel')
            conloader1 = DataLoader(conheTensorData, batch_size=self.batch_size, shuffle=True, num_workers=0,pin_memory=False)
            coniter1 = iter(conloader1)


        for epoch in range(self.epoch):
            model.train()
            batchTrAccSum     = 0.0
            batchTrLossSPSum  = 0.0
            batchTrLossMLMSum = 0.0
            batchTrLossConSum = 0.0
            timeEpochStart    = time.time()

            embedding = []
            Y_1 = []
            for batch in dataloader:

                Y, ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                     'token_type_ids':types.to(model.device),
                     'attention_mask':masks.to(model.device)}

                # forward
                CLSEmbedding_bert,logits = model(X)
                embedding.append(CLSEmbedding_bert)
                embedding1 = torch.cat(embedding, dim=0)
                Y_1.append(Y)
                Y_all = torch.cat(Y_1, dim=0)
                if self.regression:
                    lossSP = model.loss_mse(logits, Y.to(model.device))
                else:
                    lossSP = model.loss_ce(logits, Y.to(model.device))

                #mlm
                try:
                    ids, types, masks = unlabelediter.next()
                except StopIteration:
                    unlabelediter = iter(unlabeledloader)
                    ids, types, masks = unlabelediter.next()
                X_un = {'input_ids':ids.to(model.device),
                        'token_type_ids':types.to(model.device),
                        'attention_mask':masks.to(model.device)}
                mask_ids, mask_lb = mask_tokens(X_un['input_ids'].cpu(), tokenizer)#这个去tools64行！！
                X_un1 = {'input_ids':mask_ids.to(model.device),
                        'token_type_ids':X_un['token_type_ids'],
                        'attention_mask':X_un['attention_mask']}
                lossMLM = model.mlmForward(X_un1, mask_lb.to(model.device))

                try:
                    ids1, types1, masks1 = coniter1.next()
                except StopIteration:
                    coniter1 = iter(conloader1)
                    ids1, types1, masks1 = coniter1.next()
                X_con1= {'input_ids': ids1.to(model.device),
                    'token_type_ids': types1.to(model.device),
                    'attention_mask': masks1.to(model.device)}
                lossDropoutCLLoss = self.calculateDropoutCLLoss(model, X_con1, beforeBatchNorm=self.beforeBatchNorm)

                # loss
                lossTOT = lossSP + self.lambda_mlm * lossMLM +  self.lossContrastiveWeight * lossDropoutCLLoss

                # backward
                self.optimizer.zero_grad()
                lossTOT.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()

                # calculate train acc
                YTensor = Y.cpu()
                logits = logits.detach().clone()
                if torch.cuda.is_available():
                    logits = logits.cpu()
                if self.regression:
                    predictResult = torch.sigmoid(logits).numpy()
                    acc = r2_score(YTensor, predictResult)
                else:
                    logits = logits.numpy()
                    predictResult = np.argmax(logits, 1)
                    acc = accuracy_score(YTensor, predictResult)
                # accumulate statistics
                batchTrAccSum += acc
                batchTrLossSPSum += lossSP.item()


            # current epoch training done, collect data
            durationTrain         = self.round(time.time() - timeEpochStart)
            durationOverallTrain += durationTrain
            batchTrAccAvrg        = self.round(batchTrAccSum/len(dataloader))
            batchTrLossSPAvrg     = batchTrLossSPSum/len(dataloader)
            batchTrLossMLMAvrg    = batchTrLossMLMSum/len(dataloader)
            batchTrLossConAvrg = batchTrLossConSum / len(dataloader)


            valAcc, valPre, valRec, valFsc = self.valEvaluator.evaluate(model, tokenizer, mode)
            teAcc, tePre, teRec, teFsc     = self.testEvaluator.evaluate(model, tokenizer, mode)

            # display current epoch's info
            logger.info("---- epoch: %d/%d, train_time %f ----", epoch, self.epoch, durationTrain)
            logger.info("SPLoss %f, MLMLoss %f, ConLoss %f, TrainAcc %f", batchTrLossSPAvrg, batchTrLossMLMAvrg, batchTrLossConAvrg, batchTrAccAvrg)
            logger.info("ValAcc %f, Val pre %f, Val rec %f , Val Fsc %f", valAcc, valPre, valRec, valFsc)
            logger.info("TestAcc %f, Test pre %f, Test rec %f, Test Fsc %f", teAcc, tePre, teRec, teFsc)


            if self.tensorboard:
                self.writer.add_scalar('train loss', batchTrLossSPAvrg+self.lambda_mlm*batchTrLossMLMAvrg, global_step=epoch)
                self.writer.add_scalar('val acc', valAcc, global_step=epoch)
                self.writer.add_scalar('test acc', teAcc, global_step=epoch)

            # early stop
            if not self.validation:
                valAcc = -1
            if (valAcc >= valBestAcc):   # better validation result
                print("[INFO] Find a better model. Val acc: %f -> %f"%(valBestAcc, valAcc))
                valBestAcc = valAcc
                accumulateStep = 0

                # cache current model, used for evaluation later
                self.bestModelStateDict = copy.deepcopy(model.state_dict())
            else:
               accumulateStep += 1
               if accumulateStep > self.patience/2:
                   print('[INFO] accumulateStep: ', accumulateStep)
                   if accumulateStep == self.patience:  # early stop
                       logger.info('Early stop.')
                       logger.debug("Overall training time %f", durationOverallTrain)
                       logger.debug("Overall validation time %f", durationOverallVal)
                       logger.debug("best_val_acc: %f", valBestAcc)
                       break
        
        logger.info("best_val_acc: %f", valBestAcc)