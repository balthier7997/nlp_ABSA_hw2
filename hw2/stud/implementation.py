import numpy as np
from typing import List, Tuple, Dict

from model import Model
import random

#####################################
import torch
import pytorch_lightning as pl
from transformers   import AutoTokenizer

from stud.utils                import *
from stud.model_class_B        import *
from stud.model_class_AB       import * 
from stud.model_class_CD       import * 
from stud.dataset_class        import *
from stud.dataset_class_extra  import *
#####################################



def build_model_b(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements aspect sentiment analysis of the ABSA pipeline.
            b: Aspect sentiment analysis.
    """
    return StudentModel(task='b', device=device)
    #return RandomBaseline()

def build_model_ab(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline.
            a: Aspect identification.
            b: Aspect sentiment analysis.

    """
    #return RandomBaseline(mode='ab')
    return StudentModel(task='ab', device=device)

def build_model_cd(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline 
        as well as Category identification and sentiment analysis.
            c: Category identification.
            d: Category sentiment analysis.
    """
    #return RandomBaseline(mode='cd')
    return StudentModel(task='cd', device=device)


class RandomBaseline(Model):

    options_sent = [
        ('positive', 793+1794),
        ('negative', 701+638),
        ('neutral',  365+507),
        ('conflict', 39+72),
    ]

    options = [
        (0, 452),
        (1, 1597),
        (2, 821),
        (3, 524),
    ]

    options_cat_n = [
        (1, 2027),
        (2, 402),
        (3, 65),
        (4, 6),
    ]

    options_sent_cat = [
        ('positive', 1801),
        ('negative', 672),
        ('neutral',  411),
        ('conflict', 164),
    ]

    options_cat = [
        ("anecdotes/miscellaneous", 939),
        ("price", 268),
        ("food", 1008),
        ("ambience", 355),
        ("service", 248),
    ]

    def __init__(self, mode = 'b'):

        self._options_sent = [option[0] for option in self.options_sent]
        self._weights_sent = np.array([option[1] for option in self.options_sent])
        self._weights_sent = self._weights_sent / self._weights_sent.sum()

        if mode == 'ab':
            self._options = [option[0] for option in self.options]
            self._weights = np.array([option[1] for option in self.options])
            self._weights = self._weights / self._weights.sum()
        elif mode == 'cd':
            self._options_cat_n = [option[0] for option in self.options_cat_n]
            self._weights_cat_n = np.array([option[1] for option in self.options_cat_n])
            self._weights_cat_n = self._weights_cat_n / self._weights_cat_n.sum()

            self._options_sent_cat = [option[0] for option in self.options_sent_cat]
            self._weights_sent_cat = np.array([option[1] for option in self.options_sent_cat])
            self._weights_sent_cat = self._weights_sent_cat / self._weights_sent_cat.sum()

            self._options_cat = [option[0] for option in self.options_cat]
            self._weights_cat = np.array([option[1] for option in self.options_cat])
            self._weights_cat = self._weights_cat / self._weights_cat.sum()

        self.mode = mode

    def predict(self, samples: List[Dict]) -> List[Dict]:
        preds = []
        for sample in samples:
            pred_sample = {}
            words = None
            if self.mode == 'ab':
                n_preds = np.random.choice(self._options, 1, p=self._weights)[0]
                if n_preds > 0 and len(sample["text"].split(" ")) > n_preds:
                    words = random.sample(sample["text"].split(" "), n_preds)
                elif n_preds > 0:
                    words = sample["text"].split(" ")
            elif self.mode == 'b':
                if len(sample["targets"]) > 0:
                    words = [word[1] for word in sample["targets"]]
            if words:
                pred_sample["targets"] = [(word, str(np.random.choice(self._options_sent, 1, p=self._weights_sent)[0])) for word in words]
            else: 
                pred_sample["targets"] = []
            if self.mode == 'cd':
                n_preds = np.random.choice(self._options_cat_n, 1, p=self._weights_cat_n)[0]
                pred_sample["categories"] = []
                for i in range(n_preds):
                    category = str(np.random.choice(self._options_cat, 1, p=self._weights_cat)[0]) 
                    sentiment = str(np.random.choice(self._options_sent_cat, 1, p=self._weights_sent_cat)[0]) 
                    pred_sample["categories"].append((category, sentiment))
            preds.append(pred_sample)
        return preds


class StudentModel(Model):
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self, task, device) -> None:
        super().__init__()
        MODEL = "roberta-base"
        pl.seed_everything(42, workers=True)
        self.task   = task
        self.device = torch.device(device)

        if self.task == 'b':
            label_B_vocab ={NEUTRAL  : 0,
                            POSITIVE : 1,
                            NEGATIVE : 2,
                            CONFLICT : 3,
                            ABSENT   : 4}
            label_B_vocab_itos = {v : k for k, v in label_B_vocab.items()}

            tokenizer  = AutoTokenizer.from_pretrained(MODEL)
            self.hparams_b={'l_vocab'         : label_B_vocab,
                            'l_vocab_itos'    : label_B_vocab_itos,
                            'num_classes'     : len(label_B_vocab),
                            'tokenizer'       : tokenizer,
                            'dropout'         : 0.0, 
                            'learning_rate'   : 5e-5, 
                            'hidden_dim'      : 768,
                            'device'          : self.device,
                            'stopset'         : set(),
                            'bert_type'       : 'roberta-base',
                            'first'           : 'text',
                            'classifier'      : 'relu',
                            'layers'          : 3,
                            'remove_stopwords': False,
                            'dummy'           : True,
                            'custom-pooler'   : None,
                            'pooler-layer'    : None,
                            'absent_token'    : '',
                            'best_model_path' : "model/model_b/roberta/B_roberta-SeqCls_relu-3_text-first_lr=3e-05_drop=0.3_epoch=5_step=1331_macro_f1=58.50.ckpt"}
            
            self.model_b = TermPolarityModelROBERTA.load_from_checkpoint(self.hparams_b['best_model_path'], hparams=self.hparams_b)
        
        elif self.task == 'ab':

            ################################### stuff for model A ###################################
            
            label_A_vocab = { PAD_TOKEN_ROBERTA if 'roberta' in MODEL else PAD_TOKEN_BERT   : 0,
                            BEGINNING : 1,
                            INSIDE    : 2,
                            OUTSIDE   : 3}
            label_A_vocab_itos = {v : k for k, v in label_A_vocab.items()}

            tokenizer = AutoTokenizer.from_pretrained(MODEL)
            self.hparams_a={ 'l_vocab'         : label_A_vocab,
                        'l_vocab_itos'    : label_A_vocab_itos,
                        'num_classes'     : len(label_A_vocab), 
                        'tokenizer'       : tokenizer,
                        'device'          : self.device,
                        'tokenizer'       : tokenizer,
                        'dropout'         : 0.0, 
                        'learning_rate'   : 5e-5,
                        'hidden_dim'      : 768,
                        'stopset'         : set(),
                        'bert_type'       : MODEL,
                        'classifier'      : 'tanh',
                        'layers'          : 3,
                        'rm_stopwords'    : False,
                        'cls_token'       : CLS_TOKEN_ROBERTA if 'roberta' in MODEL else CLS_TOKEN_BERT,
                        'sep_token'       : SEP_TOKEN_ROBERTA if 'roberta' in MODEL else SEP_TOKEN_BERT,
                        'unk_token'       : UNK_TOKEN_ROBERTA if 'roberta' in MODEL else UNK_TOKEN_BERT,
                        'pad_token'       : PAD_TOKEN_ROBERTA if 'roberta' in MODEL else PAD_TOKEN_BERT,
                        'uncased'         : ('uncased' in MODEL),
                        'best_model_path' : "model/model_a/roberta/A_roberta-TokCls_tanh-3_lr=5e-5_drop=0.2_epoch=12_step=2040_F1_score=83.58.ckpt"}
                        #'best_model_path' : "model/model_a/roberta/roberta-TokCls_tanh-3_epoch=6_step=1098_F1_score=83.10.ckpt"}


            ################################### stuff for model B ###################################
            label_B_vocab ={NEUTRAL  : 0,
                            POSITIVE : 1,
                            NEGATIVE : 2,
                            CONFLICT : 3,
                            ABSENT   : 4}
            label_B_vocab_itos = {v : k for k, v in label_B_vocab.items()}

            tokenizer  = AutoTokenizer.from_pretrained('roberta-base')
            self.hparams_b={'l_vocab'         : label_B_vocab,
                            'l_vocab_itos'    : label_B_vocab_itos,
                            'num_classes'     : len(label_B_vocab),
                            'tokenizer'       : tokenizer,
                            'dropout'         : 0.0, 
                            'learning_rate'   : 5e-5,
                            'hidden_dim'      : 768,
                            'device'          : self.device,
                            'stopset'         : set(),
                            'bert_type'       : MODEL,
                            'first'           : 'text',
                            'classifier'      : 'relu',
                            'layers'          : 3,
                            'remove_stopwords': False,
                            'dummy'           : True,
                            'custom-pooler'   : None,
                            'pooler-layer'    : None,
                            'absent_token'    : '',
                            'best_model_path' : "model/model_b/roberta/B_roberta-SeqCls_relu-3_text-first_lr=3e-05_drop=0.3_epoch=5_step=1331_macro_f1=58.50.ckpt"}

            self.model_ab = ModelAB(hparams_a=self.hparams_a, hparams_b=self.hparams_b)
        
        else:   # task == 'cd'
            ################################### stuff for model C ###################################
            C_FILE      = "roberta/C_roberta-MultiLabel_selu-3_epoch=2_step=500_macro_f1=86.06"
            tokenizer = AutoTokenizer.from_pretrained(MODEL)
            label_C_vocab= {FOOD     : 0,
                            MISC     : 1,
                            SERVICE  : 2,
                            AMBIENCE : 3,
                            PRICE    : 4}
            label_C_vocab_itos = {v : k for k, v in label_C_vocab.items()}

            self.hparams_C={'bert_type'       : MODEL,
                            'num_classes'     : len(label_C_vocab), 
                            'l_vocab'         : label_C_vocab,
                            'l_vocab_itos'    : label_C_vocab_itos,
                            'tokenizer'       : tokenizer,
                            'dropout'         : 0.0, 
                            'learning_rate'   : 5e-5,
                            'dropout'         : 0.05,
                            'hidden_dim'      : 768,
                            'classifier'      : 'selu',
                            'layers'          : 3,
                            'device'          : device,
                            'best_model_path' : f"model/model_c/{C_FILE}.ckpt"}#


            ################################### stuff for model D ###################################
            D_FILE     = "roberta/D_roberta-SeqCls_tanh-2_text-first_epoch=19_step=1759_macro_f1=71.96"

            tokenizer = AutoTokenizer.from_pretrained(MODEL)
            label_D_vocab = {   NEUTRAL  : 0,
                                POSITIVE : 1,
                                NEGATIVE : 2,
                                CONFLICT : 3}
            label_D_vocab_itos = {v : k for k, v in label_D_vocab.items()}

            self.hparams_D={'bert_type'       : MODEL,
                            'num_classes'     : len(label_D_vocab),
                            'l_vocab'         : label_D_vocab, 
                            'l_vocab_itos'    : label_D_vocab_itos,
                            'tokenizer'       : tokenizer,
                            'dropout'         : 0.0, 
                            'learning_rate'   : 5e-5,
                            'hidden_dim'      : 768,
                            'classifier'      : 'tanh',
                            'layers'          : 2,
                            'first'           : 'text',
                            'device'          : device,#
                            'best_model_path' : f"model/model_d/{D_FILE}.ckpt"}

            self.model_cd = ModelCD(hparams_c=self.hparams_C, hparams_d=self.hparams_D)



    def predict(self, samples: List[Dict]) -> List[Dict]:
        '''
            --> !!! STUDENT: implement here your predict function !!! <--
            Args:
                - If you are doing model_b (ie. aspect sentiment analysis):
                    sentence: a dictionary that represents an input sentence as well as the target words (aspects), for example:
                        [
                            {
                                "text": "I love their pasta but I hate their Ananas Pizza.",
                                "targets": [[13, 17], "pasta"], [[36, 47], "Ananas Pizza"]]
                            },
                            {
                                "text": "The people there is so kind and the taste is exceptional, I'll come back for sure!"
                                "targets": [[4, 9], "people", [[36, 40], "taste"]]
                            }
                        ]
                - If you are doing model_ab or model_cd:
                    sentence: a dictionary that represents an input sentence, for example:
                        [
                            {
                                "text": "I love their pasta but I hate their Ananas Pizza."
                            },
                            {
                                "text": "The people there is so kind and the taste is exceptional, I'll come back for sure!"
                            }
                        ]
            Returns:
                A List of dictionaries with your predictions:
                    - If you are doing target word identification + target polarity classification:
                        [
                            {
                                "targets": [("pasta", "positive"), ("Ananas Pizza", "negative")] # A list having a tuple for each target word
                            },
                            {
                                "targets": [("people", "positive"), ("taste", "positive")] # A list having a tuple for each target word
                            }
                        ]
                    - If you are doing target word identification + target polarity classification + aspect category identification + aspect category polarity classification:
                        [
                            {
                                "targets": [("pasta", "positive"), ("Ananas Pizza", "negative")], # A list having a tuple for each target word
                                "categories": [("food", "conflict")]
                            },
                            {
                                "targets": [("people", "positive"), ("taste", "positive")], # A list having a tuple for each target word
                                "categories": [("service", "positive"), ("food", "positive")]
                            }
                        ]
        '''
        if self.task == 'b':
            
            result = self.model_b.predict_task_b(raw_data=samples, l_vocab=self.hparams_b['l_vocab'])
            self.model_b.sanitize_labels(samples, absent_token=self.hparams_b['absent_token'])
            return result
        
    
        elif self.task == 'ab':
            return self.model_ab.predict_task_ab(samples)


        else:   # task == 'cd'
            return self.model_cd.predict_task_cd(samples)
