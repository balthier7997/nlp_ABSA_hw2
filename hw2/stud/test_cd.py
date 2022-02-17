import sys
import torch
import pytorch_lightning as pl

from time         import sleep
from transformers import AutoTokenizer

from utils  import *
from dataset_class_extra import CategoryExtractionDatasetBERT
from dataset_class_extra import CategorySentimentDatasetBERT

from model_class_CD import ModelCD

import gc
gc.collect()

pl.seed_everything(42, workers=True)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")


## stuff for model C ##
NUM_LAYERS  =  3
ACT         = 'selu'
MODEL       = "roberta-base"
C_FILE      = "roberta/C_roberta-MultiLabel_selu-3_epoch=2_step=500_macro_f1=86.06"
HIDDEN_DIM  = 768
tokenizer = AutoTokenizer.from_pretrained(MODEL)
label_C_vocab= {FOOD     : 0,
                MISC     : 1,
                SERVICE  : 2,
                AMBIENCE : 3,
                PRICE    : 4}
label_C_vocab_itos = {v : k for k, v in label_C_vocab.items()}

hparams_C={ 'bert_type'       : MODEL,
            'num_classes'     : len(label_C_vocab), 
            'l_vocab'         : label_C_vocab,
            'l_vocab_itos'    : label_C_vocab_itos,
            'tokenizer'       : tokenizer,
            'device'          : device,
            'dropout'         : 0.05,
            'classifier'      : ACT,
            'layers'          : NUM_LAYERS,
            'hidden_dim'      : HIDDEN_DIM,
            'best_model_path' : f"model/model_c/{C_FILE}.ckpt"}#


## stuff for model D ##
MODEL      = "roberta-base"
ACT        = 'tanh'
NUM_LAYERS =  2
FIRST      = 'text'
D_FILE     = "roberta/D_roberta-SeqCls_tanh-2_text-first_epoch=19_step=1759_macro_f1=71.96"
HIDDEN_DIM  = 768

tokenizer = AutoTokenizer.from_pretrained(MODEL)
label_D_vocab = { NEUTRAL  : 0,
                  POSITIVE : 1,
                  NEGATIVE : 2,
                  CONFLICT : 3}
label_D_vocab_itos = {v : k for k, v in label_D_vocab.items()}

hparams_D={ 'bert_type'       : MODEL,
            'num_classes'     : len(label_D_vocab),
            'l_vocab'         : label_D_vocab, 
            'l_vocab_itos'    : label_D_vocab_itos,
            'tokenizer'       : tokenizer,
            'dropout'         : 0.05,
            'classifier'      : ACT,
            'layers'          : NUM_LAYERS,
            'hidden_dim'      : HIDDEN_DIM,
            'first'           : FIRST,
            'device'          : device,#
            'best_model_path' : f"model/model_d/{D_FILE}.ckpt"}

ModelCD = ModelCD(hparams_c=hparams_C, hparams_d=hparams_D)

### TESTING ###

DEVEL_FILE_R = 'data/restaurants_dev.json'
devel_files = [DEVEL_FILE_R]

flag = sys.argv[1].lower()

if flag == 'c':
    test_dataset_C = CategoryExtractionDatasetBERT.test_dataset_builder(devel_files)#, tokenizer)
    preds_task_c   = ModelCD.predict_task_c(test_dataset_C)
    sleep(0.5)
    print("\nTask C evaluation")
    eval_task_C(data=test_dataset_C, preds_c=preds_task_c, display=True)


if flag == 'd':
    test_dataset_D = CategorySentimentDatasetBERT.test_dataset_builder(devel_files)
    preds_task_d   = ModelCD.predict_task_d(raw_data=test_dataset_D)
    sleep(0.5)
    print("\nTask D evaluation")
    eval_task_D(data=test_dataset_D, preds_d=preds_task_d, display=True)


if flag == 'cd':
    test_dataset_C = CategoryExtractionDatasetBERT.test_dataset_builder(devel_files)#, tokenizer)
    test_dataset_D = CategorySentimentDatasetBERT.test_dataset_builder(devel_files)
    preds_task_cd  = ModelCD.predict_task_cd(data_task_c=test_dataset_C)
    sleep(2)
    print("\nTask CD evaluation")
    eval_task_D(data=test_dataset_D, preds_d=preds_task_cd, display=True)


