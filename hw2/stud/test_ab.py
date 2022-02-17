import sys
import torch
import pytorch_lightning as pl

from time         import sleep
from transformers import AutoTokenizer, RobertaTokenizer

from utils          import *
from dataset_class  import TermIdentificationDatasetBERT
from dataset_class  import TermPolarityDatasetBERT

from model_class_AB import ModelAB

import gc
gc.collect()

pl.seed_everything(42, workers=True)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")

################################### stuff for model A ###################################

ACT        = 'tanh'
NUM_LAYERS = 3
MODEL      = "roberta-base"
HIDDEN_DIM       = 768
REMOVE_STOPWORDS = False

label_A_vocab = { PAD_TOKEN_ROBERTA if 'roberta' in MODEL else PAD_TOKEN_BERT   : 0,
                  BEGINNING : 1,
                  INSIDE    : 2,
                  OUTSIDE   : 3}
label_A_vocab_itos = {v : k for k, v in label_A_vocab.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL)
hparams_A={ 'l_vocab'         : label_A_vocab,
            'l_vocab_itos'    : label_A_vocab_itos,
            'num_classes'     : len(label_A_vocab), 
            'tokenizer'       : tokenizer,
            'device'          : device,
            'dropout'         : 0.05,
            'stopset'         : set(),
            'bert_type'       : MODEL,
            'classifier'      : ACT,
            'layers'          : NUM_LAYERS,
            'hidden_dim'      : HIDDEN_DIM,
            'rm_stopwords'    : REMOVE_STOPWORDS,
            'cls_token'       : CLS_TOKEN_ROBERTA if 'roberta' in MODEL else CLS_TOKEN_BERT,
            'sep_token'       : SEP_TOKEN_ROBERTA if 'roberta' in MODEL else SEP_TOKEN_BERT,
            'unk_token'       : UNK_TOKEN_ROBERTA if 'roberta' in MODEL else UNK_TOKEN_BERT,
            'pad_token'       : PAD_TOKEN_ROBERTA if 'roberta' in MODEL else PAD_TOKEN_BERT,
            'uncased'         : ('uncased' in MODEL),
            'best_model_path' : "model/model_a/roberta/A_roberta-TokCls_tanh-3_lr=5e-5_drop=0.2_epoch=12_step=2040_F1_score=83.58.ckpt"}


################################### stuff for model B ###################################
label_B_vocab ={NEUTRAL  : 0,
                POSITIVE : 1,
                NEGATIVE : 2,
                CONFLICT : 3,
                ABSENT   : 4}
label_B_vocab_itos = {v : k for k, v in label_B_vocab.items()}

ACT              = 'relu'
NUM_LAYERS       = 3
MODEL            = "roberta-base"
FIRST            = 'text'
POOLING          = None
DUMMY_LABEL      = True
ABSENT_TOKEN     = ''
POOLER_LAYER     = None
REMOVE_STOPWORDS = False
HIDDEN_DIM       = 768
tokenizer  = RobertaTokenizer.from_pretrained(MODEL)
hparams_B={ 'l_vocab'         : label_B_vocab,
            'l_vocab_itos'    : label_B_vocab_itos,
            'num_classes'     : len(label_B_vocab),
            'tokenizer'       : tokenizer,
            'dropout'         : 0.0, 
            'device'          : device,
            'stopset'         : set(),
            'bert_type'       : MODEL,
            'first'           : FIRST,
            'model'           : MODEL,
            'classifier'      : ACT,
            'layers'          : NUM_LAYERS,
            'remove_stopwords': REMOVE_STOPWORDS,
            'dummy'           : DUMMY_LABEL,
            'custom-pooler'   : POOLING,
            'pooler-layer'    : POOLER_LAYER,
            'absent_token'    : ABSENT_TOKEN,
            'hidden_dim'      : HIDDEN_DIM,
            'best_model_path' : "model/model_b/roberta/B_roberta-SeqCls_relu-3_text-first_lr=3e-05_drop=0.3_epoch=5_step=1331_macro_f1=58.50.ckpt"}
            #'best_model_path' : "model/model_b/roberta/roberta-SeqCls_relu-3_text-first_epoch=4_step=1109_macro_f1=56.90.ckpt"}

modelAB = ModelAB(hparams_a=hparams_A, hparams_b=hparams_B)
#modelAB = ModelAB_noDummy(hparams_a=hparams_A, hparams_b=hparams_B)

### TESTING ###
devel_files = [DEVEL_FILE_R, DEVEL_FILE_L]

flag = sys.argv[1].lower()

if flag == 'a':
    test_dataset_A = TermIdentificationDatasetBERT( input_files=devel_files, 
                                                    tokenizer=tokenizer, 
                                                    stopset=hparams_A['stopset'], 
                                                    rm_stopwords=hparams_A['rm_stopwords'],
                                                    roberta=('roberta' in hparams_A['bert_type']))
    preds_task_a = modelAB.predict_task_a(test_dataset_A)
    sleep(0.5)
    print("\nTask A evaluation")
    eval_task_A(test_dataset_A, preds_task_a, display=True)


if flag == 'b':
    test_dataset_B = TermPolarityDatasetBERT.test_dataset_builder(input_files=devel_files)
    preds_task_b = modelAB.predict_task_b(raw_data=test_dataset_B)
    sleep(0.5)
    print("\nTask B evaluation")
    eval_task_B(data=test_dataset_B, preds_b=preds_task_b)


if flag == 'ab':
    test_dataset_A = TermIdentificationDatasetBERT( input_files=devel_files, 
                                                    tokenizer=tokenizer, 
                                                    stopset=hparams_A['stopset'], 
                                                    rm_stopwords=hparams_A['rm_stopwords'],
                                                    roberta=('roberta' in hparams_A['bert_type']))
    test_dataset_B = TermPolarityDatasetBERT.test_dataset_builder(input_files=devel_files)
    preds_task_ab = modelAB.predict_task_ab(test_dataset_A)
    sleep(2)
    eval_task_B(data=test_dataset_B, preds_b=preds_task_ab, display=True)


