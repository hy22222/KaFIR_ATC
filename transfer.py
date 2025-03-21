import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

import torch
import torch.optim as optim
import argparse
import time
import copy
from transformers import BertModel, BertTokenizer
import random
import torch.utils.data as Data
from utils.models import IntentBERT
from utils.IntentDataset import IntentDataset
from utils.Trainer import TransferTrainer,MLMOnlyTrainer
from utils.Evaluator import FewShotEvaluator
from utils.commonVar import *
from utils.printHelper import *
from utils.tools import *
from utils.Logger import logger
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parseArgument():
    # ==== parse argument ====
    parser = argparse.ArgumentParser(description='Train IntentBERT')

    # ==== model ====
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--mode', default='multi-class',
                        help='Choose from multi-class')
    parser.add_argument('--tokenizer', default='bert-base-Chinese',
                        help="Name of tokenizer")
    parser.add_argument('--LMName', default='bert-base-Chinese',
                        help='Name for models and path to saved model，bert-base-uncased,bert-base-chinese,distilbert-base-uncased')

    parser.add_argument('--beforeBatchNorm', help='Use the features before batch norm to infer or not', action="store_false")
    parser.add_argument('-simTemp', default=0.05, type=float, help="The temperature of similarity during contrastive learning.")
    parser.add_argument('--lossContrastiveWeight', type=float, default=1)

    # ==== dataset ====
    parser.add_argument('--dataDir',default='train_dataset,val_dataset,test_dataset,test_he',
                        help="Dataset names included in this experiment and separated by comma. "
                        "For example:'OOS,bank77,hwu64,acc,train_dataset,val_dataset'")
    parser.add_argument('--sourceDomain',default='all_traindata',
                        help="Source domain names and separated by comma.[[acc_traindata]] "
                        "For example:'travel,banking,home'")
    parser.add_argument('--valDomain',default='all_valdata',
                        help='Validation domain names and separated by comma.[[app_valdata]]')
    parser.add_argument('--targetDomain',default='all_testdata',
                        help='Target domain names and separated by comma[[acc_data]]')
    parser.add_argument('--conDomainhe',default='con_datahe',
                        help='con domain names and separated by comma[[con_data]]')


    # ==== evaluation task ====
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=15)
    parser.add_argument('--query', type=int, default=5)
    parser.add_argument('--clsFierName', default='Linear',
   help="Classifer name for few-shot evaluation"
                        "Choose from Linear|SVM|NN|Cosine|MultiLabel")

    # ==== optimizer ====
    parser.add_argument('--optimizer', default='Adam',
                        help='Choose from SGD|Adam')
    parser.add_argument('--learningRate', type=float, default=2e-5)
    parser.add_argument('--weightDecay', type=float, default=0)

    # ==== training arguments ====
    parser.add_argument('--disableCuda', action="store_true")
    parser.add_argument('--validation', action="store_true")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)#
    parser.add_argument('--taskNum', type=int, default=50)
    parser.add_argument('--patience', type=int, default=3,
                        help="Early stop when performance does not go better")
    #==== MLM参数 ====
    parser.add_argument('--mlm', default='store_false',
                        help="If use mlm as auxiliary loss")
    parser.add_argument('--lambda_mlm', type=float, default=7,
                        help="The weight for mlm loss[[4]]")
    parser.add_argument('--mlm_data', default='target', type=str,
                        help="Data for mlm. Choose from target|source")
    parser.add_argument('--shuffle_mlm', action="store_true")
    parser.add_argument('--shuffle', action="store_false")
    parser.add_argument('--regression', action="store_true",
                        help="If the pretrain task is a regression task")
    # === con ===
    parser.add_argument('--con', default='store_false',
                        help="If use con as auxiliary loss")

    # ==== other things ====
    parser.add_argument('--loggingLevel', default='INFO',
                        help="python logging level")
    parser.add_argument('--saveModel', action='store_true',
                        help="Whether to save pretrained model")
    parser.add_argument('--saveName', default='none',
                        help="Specify a unique name to save your model"
                        "If none, then there will be a specific name controlled by how the model is trained")
    parser.add_argument('--tensorboard', action='store_true',
                        help="Enable tensorboard to log training and validation accuracy")
    parser.add_argument('--batchMonitor', type=int, help='for how many batches to monitor the validation performance', default=50)
    args = parser.parse_args()

    return args

def main():
    # ======= process arguments ======
    args = parseArgument()
    print(args)

    if not args.saveModel:
        logger.info("The model will not be saved after training!")

    # ==== setup logger ====
    if args.loggingLevel == LOGGING_LEVEL_INFO:
        loggingLevel = logging.INFO
    elif args.loggingLevel == LOGGING_LEVEL_DEBUG:
        loggingLevel = logging.DEBUG
    else:
        raise NotImplementedError("Not supported logging level %s", args.loggingLevel)
    logger.setLevel(loggingLevel)

    # ==== set seed ====
    set_seed(args.seed)

    # ======= process data ======
    # tokenizer
    tok = BertTokenizer.from_pretrained("./BERT_model/")
    # load raw dataset
    logger.info(f"Loading data from {args.dataDir}")
    dataset = IntentDataset(regression=args.regression)
    dataset.loadDataset(splitName(args.dataDir))
    dataset.tokenize(tok)
    # spit data into training, validation and testing(read-back pattern)
    logger.info("----- Training Data -----")
    trainData = dataset.splitDomain(splitName(args.sourceDomain), regression=args.regression)
    logger.info("----- Validation Data -----")
    valData = dataset.splitDomain(splitName(args.valDomain))
    logger.info("----- Testing Data -----")
    testData = dataset.splitDomain(splitName(args.targetDomain))
    conDatahe = dataset.splitDomain(splitName(args.conDomainhe))
    # shuffle word order
    if args.shuffle:
        trainData.shuffle_words()

    # ======= prepare model ======
    # initialize model
    modelConfig = {}
    modelConfig['device'] = torch.device('cuda:0' if not args.disableCuda else 'cpu')
    if args.regression:
        modelConfig['clsNumber'] = 1
    else:
        modelConfig['clsNumber'] = trainData.getLabNum()
    modelConfig['LMName'] = args.LMName
    modelConfig['simTemp'] = args.simTemp
    model = IntentBERT(modelConfig)
    logger.info("----- IntentBERT initialized -----")

    # setup validator
    valParam = {"evalTaskNum": args.taskNum, "clsFierName": args.clsFierName, "multi_label": False}
    valTaskParam = {"way":args.way, "shot":args.shot, "query":args.query}
    validator = FewShotEvaluator(valParam, valTaskParam, valData)
    tester = FewShotEvaluator(valParam, valTaskParam, testData)

    # setup trainer
    optimizer = None
    if args.optimizer == OPTER_ADAM:
        optimizer = optim.Adam(model.parameters(), lr=args.learningRate, weight_decay=args.weightDecay)
    elif args.optimizer == OPTER_SGD:
        optimizer = optim.SGD(model.parameters(), lr=args.learningRate, weight_decay=args.weightDecay)
    else:
        raise NotImplementedError("Not supported optimizer %s"%(args.optimizer))

    if args.mlm and args.mlm_data == "target":
        # args.validation = False
        args.validation = True
    trainingParam = {"epoch"      : args.epochs, \
                     "batch"      : args.batch_size, \
                     "validation" : args.validation, \
                     "patience"   : args.patience, \
                     "tensorboard": args.tensorboard, \
                     "mlm"        : args.mlm, \
                     "lambda mlm" : args.lambda_mlm, \
                     "regression" : args.regression,\
                     "con"        : args.con,\
                     "beforeBatchNorm": args.beforeBatchNorm, \
                     "lossContrastiveWeight": args.lossContrastiveWeight, \
                     "batchMonitor": args.batchMonitor, \
                     "wandb": None, \
                     "wandbProj": None, \
                     "wandbRunName": None, \
                     "wandbConfig": None
                     }

    unlabeledData = None
    if args.mlm_data == "source":
        unlabeledData = copy.deepcopy(trainData)
    elif args.mlm_data == "target":
        unlabeledData = copy.deepcopy(testData)
    if args.shuffle_mlm:
        unlabeledData.shuffle_words()

    condatahe = copy.deepcopy(conDatahe)
    trainer = TransferTrainer(trainingParam, optimizer, trainData, unlabeledData, condatahe, validator, tester)
    # train
    trainer.train(model, tok)
    # load best model
    bestModelStateDict = trainer.getBestModelStateDict()
    model.load_state_dict(bestModelStateDict)

    # evaluate once more to show results
    tester.evaluate(model, tok, args.mode, logLevel='INFO')

    logger.info(args)
    logger.info(time.asctime())

if __name__ == "__main__":
    main()
