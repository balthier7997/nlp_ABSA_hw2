import torch
import json

from pprint import pprint
from typing import List, Dict
from sklearn.metrics import f1_score
from collections     import Counter, OrderedDict
from transformers    import BertModel, BertPreTrainedModel, RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

''' Common constants '''

TRAIN_FILE_R = 'data/restaurants_train.json'
DEVEL_FILE_R = 'data/restaurants_dev.json'
TRAIN_FILE_L = 'data/laptops_train.json'
DEVEL_FILE_L = 'data/laptops_dev.json'

CLS_TOKEN_BERT = '[CLS]'
SEP_TOKEN_BERT = '[SEP]'
PAD_TOKEN_BERT = '[PAD]'
UNK_TOKEN_BERT = '[UNK]'

CLS_TOKEN_ROBERTA = '<s>'
SEP_TOKEN_ROBERTA = '</s>'
UNK_TOKEN_ROBERTA = '<unk>'
PAD_TOKEN_ROBERTA = '<pad>'

BEGINNING = 'Beginning'
INSIDE    = 'Inside'
OUTSIDE   = 'Outside'

NEUTRAL   = 'neutral'
POSITIVE  = 'positive'
NEGATIVE  = 'negative'
CONFLICT  = 'conflict'
ABSENT    = 'absent'

FOOD      = 'food'
MISC      = 'anecdotes/miscellaneous'
SERVICE   = 'service'
AMBIENCE  = 'ambience'
PRICE     = 'price'


""" Evaluation utils """


def compute_f1_seqtag_score(model, test_dataset):
    """
    Function to compute the micro F1, macro F1 and per-class F1 score on the Sequence Tagging problem (PoS, NER, ...) \n
    All models implement their own version of it
    """
    model.freeze()
    preds_vector = []
    labels_vector = []
    for elem in test_dataset:
        inputs = elem["idxs_vector"].unsqueeze(1).cpu()
        labels = elem["terms_vector"].cpu()
        _, preds = model(inputs)
        preds  = preds.view( -1)
        labels = labels.view(-1)

        indices = labels != test_dataset.l_vocab[test_dataset.pad_token]
        valid_predictions = preds [indices]
        valid_labels      = labels[indices]
        
        preds_vector.extend(valid_predictions.tolist())
        labels_vector.extend(valid_labels.tolist())
    
    micro_f1 = f1_score(labels_vector, preds_vector, average="micro", zero_division=0)
    macro_f1 = f1_score(labels_vector, preds_vector, average="macro", zero_division=0)
    class_f1 = f1_score(labels_vector, preds_vector, labels=range(len(test_dataset.l_vocab)), average=None, zero_division=0)
    model.unfreeze()
    return {"micro_f1" : micro_f1,
            "macro_f1" : macro_f1, 
            "class_f1" : class_f1}

def eval_task_A(data: List[Dict], preds_a: List[Dict], display: bool = False, debug: bool = False, tofile: bool = False) -> float:
    """
    **Code adapted from `evaluation.py`**                       \n 
    Function to evaluate the predictions for task A             \n
    Used by `utils.py` functions and `model_class.py` functions
    """
    filepath="/media/nemo/DATA/uni/nlp-hw2/hw2/stud/results/results_A.txt"
    tp = 0
    fp = 0
    fn = 0
    for label, preds in zip (data, preds_a):
        #print("ATTENTION: THERE?S A .lower() IN THE EVALUATION")
        #gold_terms = {gold[1].lower()  for gold in label["targets"]}
        gold_terms = {gold[1]  for gold in label["targets"]}
        pred_terms = {pred[0]  for pred in preds["targets"]}
        if debug:
            if pred_terms != gold_terms:
                print()
                print("preds", pred_terms)
                print("golds", gold_terms)
        tp += len(pred_terms & gold_terms)
        fp += len(pred_terms - gold_terms)
        fn += len(gold_terms - pred_terms)

    precision = 100 * tp / (tp + fp)                            if tp + fp > 0 else 0
    recall    = 100 * tp / (tp + fn)                            if tp + fn > 0 else 0
    f1_score  = 2 * precision * recall / (precision + recall)   if precision + recall > 0 else 0
    if display:
        print()
        print("TP\tFP\tFN")
        print(f"{tp}\t{fp}\t{fn}")
        print(f"PRECISION: {precision:.2f}%")
        print(f"RECALL:    {recall   :.2f}%")
        print(f"F1-SCORE:  {f1_score :.2f}%")
    if tofile:
        print("TP\tFP\tFN", file=open(filepath, 'a'))
        print(f"{tp}\t{fp}\t{fn}", file=open(filepath, 'a'))
        print(f"PRECISION: {precision:.2f}%", file=open(filepath, 'a'))
        print(f"RECALL:    {recall   :.2f}%", file=open(filepath, 'a'))
        print(f"F1-SCORE:  {f1_score :.2f}%", file=open(filepath, 'a'))
    return f1_score

def print_seqtag_results(macro_f1:float, micro_f1:float, class_f1:float, l_vocab_itos: Dict):
    """
    Prints the summary of the evaluation on the Sequence Tagging problem
    """
    print()
    print(f">> MACRO F1 sequence tagging\t{macro_f1*100:.2f}%")
    print(f">> micro F1 sequence tagging\t{micro_f1*100:.2f}%")
    print(">> Per class F1 score on sequence tagging:")
    for label, f1 in enumerate(class_f1):
        print(f"\t{label} {l_vocab_itos[label]}\t{f1*100:.2f}%") if label != 0 else 0

def eval_task_B(data: List[Dict], preds_b: List[Dict], display: bool = False, tofile: bool = False) -> float:
    """
    **Code adapted from `evaluation.py`**                       \n 
    Function to evaluate the predictions for task B             \n
    Used by `utils.py` functions and `model_class.py` functions
    """
    filepath = "/media/nemo/DATA/uni/nlp-hw2/hw2/stud/results/results_B.txt"
    scores = {}
    sentiment_types = ["positive", "negative", "neutral", "conflict"]
    scores = {sent: {"tp": 0, "fp": 0, "fn": 0} for sent in sentiment_types + ["ALL"]}

    for label, pred in zip(data, preds_b):
        print("\n\nExamining", label)           if display else 0
        for sentiment in sentiment_types:
            gold_sent = {(term_pred[1], term_pred[2]) for term_pred in label["targets"] if term_pred[2] == sentiment}
            pred_sent = {(term_pred[0], term_pred[1]) for term_pred in pred["targets"]  if term_pred[1] == sentiment}
            print(">> Examining", sentiment)    if display else 0
            print("\t>> gold sent", gold_sent)  if display else 0
            print("\t>> pred sent", pred_sent)  if display else 0
            scores[sentiment]["tp"] += len(pred_sent & gold_sent)
            scores[sentiment]["fp"] += len(pred_sent - gold_sent)
            scores[sentiment]["fn"] += len(gold_sent - pred_sent)

    scores = compute_scores(scores, types_list=sentiment_types)
    f1_summary(scores=scores, types_list=sentiment_types, display=display, filepath=None)
    if tofile:
        f1_summary(scores=scores, types_list=sentiment_types, display=display, filepath=filepath)
    return scores["ALL"]["Macro_f1"], scores["ALL"]['f1']

def eval_task_C(data: List[Dict], preds_c: List[Dict], display: bool = False, tofile: bool = False, debug: bool = False) -> float:
    """
    **Code adapted from `evaluation.py`**                       \n 
    Function to evaluate the predictions for task C             \n
    Used by `utils.py` functions and `model_class.py` functions
    """
    filepath = "/media/nemo/DATA/uni/nlp-hw2/hw2/stud/results/results_C.txt"
    scores = {}
    category_types = ["anecdotes/miscellaneous", "price", "food", "ambience", "service"]
    scores = {sent: {"tp": 0, "fp": 0, "fn": 0} for sent in category_types + ["ALL"]}
    for label, pred in zip(data, preds_c):
        print("\n\nExamining", label)           if debug else 0
        for category in category_types:
            gold_cat = {(cat_pred[0]) for cat_pred in label["categories"] if cat_pred[0] == category}
            pred_cat = {(cat_pred[0]) for cat_pred in  pred["categories"] if cat_pred[0] == category}
            print(">> Examining", category)     if debug else 0
            print("\t>> gold cat", gold_cat)    if debug else 0
            print("\t>> pred cat", pred_cat)    if debug else 0
            
            scores[category]["tp"] += len(pred_cat & gold_cat)
            scores[category]["fp"] += len(pred_cat - gold_cat)
            scores[category]["fn"] += len(gold_cat - pred_cat)

    # Compute per category Precision / Recall / F1
    scores = compute_scores(scores, types_list=category_types)

    f1_summary(scores=scores, types_list=category_types, display=display, filepath=None)
    if tofile:
        f1_summary(scores=scores, types_list=category_types, display=display, filepath=filepath)

    return scores["ALL"]["Macro_f1"], scores["ALL"]["f1"]

def eval_task_D(data: List[Dict], preds_d: List[Dict], display: bool = False, tofile: bool = False, debug: bool = False) -> float:
    """
    **Code adapted from `evaluation.py`**                       \n 
    Function to evaluate the predictions for task B             \n
    Used by `utils.py` functions and `model_class.py` functions
    """
    filepath = "/media/nemo/DATA/uni/nlp-hw2/hw2/stud/results/results_D.txt"
    scores = {}
    sentiment_types = ["positive", "negative", "neutral", "conflict"]
    scores = {sent: {"tp": 0, "fp": 0, "fn": 0} for sent in sentiment_types + ["ALL"]}

    for label, pred in zip(data, preds_d):
        print("\n\nExamining", label)           if debug else 0
        for sentiment in sentiment_types:
            pred_sent = {(term_pred[0], term_pred[1]) for term_pred in  pred["categories"] if term_pred[1] == sentiment}
            gold_sent = {(term_pred[0], term_pred[1]) for term_pred in label["categories"] if term_pred[1] == sentiment}
            print(">> Examining", sentiment)    if debug else 0
            print("\t>> gold sent", gold_sent)  if debug else 0
            print("\t>> pred sent", pred_sent)  if debug else 0
            scores[sentiment]["tp"] += len(pred_sent & gold_sent)
            scores[sentiment]["fp"] += len(pred_sent - gold_sent)
            scores[sentiment]["fn"] += len(gold_sent - pred_sent)

    scores = compute_scores(scores, types_list=sentiment_types)
    f1_summary(scores=scores, types_list=sentiment_types, display=display, filepath=None)
    if tofile:
        f1_summary(scores=scores, types_list=sentiment_types, display=display, filepath=filepath)
    return scores["ALL"]["Macro_f1"], scores["ALL"]['f1']

# common to B,C,D evaluations
def compute_scores(scores, types_list):
    # Compute per category Precision / Recall / F1
    for sent_type in scores.keys():
        if scores[sent_type]["tp"]:
            scores[sent_type]["pr"] = 100 * scores[sent_type]["tp"] / (scores[sent_type]["fp"] + scores[sent_type]["tp"])
            scores[sent_type]["re"] = 100 * scores[sent_type]["tp"] / (scores[sent_type]["fn"] + scores[sent_type]["tp"])
        else:
            scores[sent_type]["pr"], scores[sent_type]["re"] = 0, 0

        if not scores[sent_type]["pr"] + scores[sent_type]["re"] == 0:
            scores[sent_type]["f1"] = 2*scores[sent_type]["pr"]*scores[sent_type]["re"] / (scores[sent_type]["pr"]+scores[sent_type]["re"])
        else:
            scores[sent_type]["f1"] = 0
    
    # Compute micro F1 Scores
    tp = sum([scores[sent_type]["tp"] for sent_type in types_list])
    fp = sum([scores[sent_type]["fp"] for sent_type in types_list])
    fn = sum([scores[sent_type]["fn"] for sent_type in types_list])

    if tp:
        precision = 100 * tp / (tp + fp)
        recall    = 100 * tp / (tp + fn)
        f1        = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["pr"] = precision
    scores["ALL"]["re"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = sum([scores[sent]["f1"] for sent in types_list])/len(types_list)
    scores["ALL"]["Macro_p"]  = sum([scores[sent]["pr"] for sent in types_list])/len(types_list)
    scores["ALL"]["Macro_r"]  = sum([scores[sent]["re"] for sent in types_list])/len(types_list)

    return scores

# common to B,C,D evaluations
def f1_summary(scores, types_list, display: bool = False, filepath=None):
    if filepath is not None:
        if display:
            for sent_type in types_list:
                print("{}:    \tTP: {:>3};\tFP: {:>3};\tFN: {:>3};\tprec: {:.2f};\trec: {:.2f};\tf1: {:.2f};\t{:>3}".format(
                    'misc' if sent_type == 'anecdotes/miscellaneous' else sent_type,
                    scores[sent_type]["tp"],
                    scores[sent_type]["fp"],
                    scores[sent_type]["fn"],
                    scores[sent_type]["pr"],
                    scores[sent_type]["re"],
                    scores[sent_type]["f1"],
                    scores[sent_type]["tp"] + scores[sent_type]["fp"]), file=open(filepath, 'a')) 
            print("-------------------------------------------------------------------------------------------------------------------", file=open(filepath, 'a'))
            print("ALL\t\t\t\tTP: {:>3};\tFP: {:>3};\tFN: {:>3};\tprec: {:.2f};\trec: {:.2f};\tf1: {:.2f}; (micro)".format(
                scores["ALL"]["tp"], 
                scores["ALL"]["fp"], 
                scores["ALL"]["fn"],
                scores["ALL"]["pr"],
                scores["ALL"]["re"],
                scores["ALL"]["f1"]), file=open(filepath, 'a'))
            print("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tprec: {:.2f};\trec: {:.2f};\tf1: {:.2f}; (Macro)\n".format(
                    scores["ALL"]["Macro_p"],
                    scores["ALL"]["Macro_r"],
                    scores["ALL"]["Macro_f1"]), file=open(filepath, 'a'))
    else:
        if display:
            for sent_type in types_list:
                print("{}:   \tTP: {:>3};\tFP: {:>3};\tFN: {:>3};\tprec: {:.2f};\trec: {:.2f};\tf1: {:.2f};\t{:>3}".format(
                    'misc' if sent_type == 'anecdotes/miscellaneous' else sent_type,
                    scores[sent_type]["tp"],
                    scores[sent_type]["fp"],
                    scores[sent_type]["fn"],
                    scores[sent_type]["pr"],
                    scores[sent_type]["re"],
                    scores[sent_type]["f1"],
                    scores[sent_type]["tp"] + scores[sent_type]["fp"])) 
            print("-------------------------------------------------------------------------------------------------------------------")
            print("ALL\t\tTP: {:>3};\tFP: {:>3};\tFN: {:>3};\tprec: {:.2f};\trec: {:.2f};\tf1: {:.2f}; (micro)".format(
                scores["ALL"]["tp"], 
                scores["ALL"]["fp"], 
                scores["ALL"]["fn"],
                scores["ALL"]["pr"],
                scores["ALL"]["re"],
                scores["ALL"]["f1"]))
            print("\t\t\t\t\t\t\t\tprec: {:.2f};\trec: {:.2f};\tf1: {:.2f}; (Macro)\n".format(
                    scores["ALL"]["Macro_p"],
                    scores["ALL"]["Macro_r"],
                    scores["ALL"]["Macro_f1"]))
        


""" Model classifier utils """

# Mish activation function
## from https://github.com/digantamisra98/Mish/blob/b5f006660ac0b4c46e2c6958ad0301d7f9c59651/Mish/Torch/mish.py
@torch.jit.script
def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))
  
class Mish(torch.nn.Module):
    def forward(self, x):
        return mish(x)


# Replace BERT's default classification with a custom one
def build_custom_cls_head(activation, num_layers, hidden_dim, num_labels, dropout_pr):
    """
    Builds a Sequential consisting of 2 or 3 Linear layers (with dropout) and a specific activation function
    """
    if activation   == 'relu':
        act_fn = torch.nn.ReLU()
    elif activation == 'gelu':
        act_fn = torch.nn.GELU()
    elif activation == 'selu':
        act_fn = torch.nn.SELU()
    elif activation == 'tanh':
        act_fn = torch.nn.Tanh()
    elif activation == 'mish':
        act_fn = Mish()
    else:
        raise NotImplementedError

    if num_layers == 2:
        classifier = torch.nn.Sequential(   torch.nn.Dropout(dropout_pr),
                                            torch.nn.Linear(768, hidden_dim),
                                            act_fn,
                                            torch.nn.Dropout(dropout_pr),
                                            torch.nn.Linear(hidden_dim, num_labels))
    elif num_layers == 3:
        classifier = torch.nn.Sequential(   torch.nn.Dropout(dropout_pr),
                                            torch.nn.Linear(768, hidden_dim),
                                            act_fn,
                                            torch.nn.Dropout(dropout_pr),
                                            torch.nn.Linear(hidden_dim, hidden_dim//2),
                                            act_fn,
                                            torch.nn.Dropout(dropout_pr),
                                            torch.nn.Linear(hidden_dim//2, num_labels))
    else:
        raise NotImplementedError
    
    return classifier

# RoBERTa needs its own stuff cause why not
class CustomRobertaClassificationHead(torch.nn.Module):
    """
    Classification head for RobertaForSequenceClassification models, cause apparently if I want to put a custom classifier the huggingface code doens't work\n
    In particular, it does'nt use the CLS token but all tokens for classification, so I just override the forward to use the CLS and rewrite the classifier
    """
    def __init__(self, activation, num_layers, hidden_dim, num_labels, dropout_pr):
        super().__init__()
        self.cls = build_custom_cls_head(activation=activation,
                                        num_layers=num_layers,
                                        hidden_dim=hidden_dim,
                                        num_labels=num_labels,
                                        dropout_pr=dropout_pr)

    def forward(self, features, **kwargs):
        return self.cls(features[:, 0, :]) # this is to take only the CLS token 


# Pooling implemented for Task B
class CustomPooling(torch.nn.Module):
    """
    Class to implement custom pooling at end of the BERT module\n
    Four pooling strategies implemented:
    - concat-mean : concatenates the mean of term and the mean of text
    - concat-max  : concatenates the max of term and the max of text
    - reduce-mean : takes the mean of the term and text
    - reduce-max  : takes the max of the term and text
    It always ignore [CLS], [SEP], [PAD] token\n
    The layer at which the computations are made can be specified as:
    - int       (as -1, so take only the last layer to perform the above computations)
    - interval  (as -4:-1, so take the average of these layers' representations and then perform computations)
    """
    def __init__(self, pooling='concat-mean', layer=-1, device='cuda'):
        super(CustomPooling, self).__init__()
        self.pooling    = pooling
        self.pool_layer = layer
        self.device     = device
    
    def forward(self, bert_out, input_ids, attention_mask, token_type_ids):
        if len(self.pool_layer) == 2:
            if self.pool_layer[1] == -1:
                hidden_states = bert_out['hidden_states'][self.pool_layer[0]:]
            else:
                hidden_states = bert_out['hidden_states'][self.pool_layer[0]:self.pool_layer[1]+1]
            hidden_states = torch.mean(torch.stack(hidden_states), dim=0)
        else: 
            hidden_states = bert_out['hidden_states'][self.pool_layer[0]]

        if self.pooling != 'cls':
            aux = []
            batch_size, seq_len, _ = hidden_states.size()
            # loop on the batch size
            for batch_id in range(batch_size):
                mask = attention_mask[batch_id]
                inps = input_ids[     batch_id]
                outs = hidden_states[ batch_id]
                typs = token_type_ids[batch_id]
                
                if 'concat' in self.pooling:
                    term = []       # assumption: first is the term, then the text
                    text = []       # only for the names tho, doesn't change the logic of the pooling
                    for i in range(seq_len):
                        if inps[i] == 102 or inps[i] == 101:    # skip CLS and SEP token
                            continue
                        if inps[i] == 0:                        # exit on first encounter of PAD token
                            break
                        if mask[i] == 1:                        # not masked --> not padding
                            if typs[i] == 0:                    # token type 0 --> first sentence
                                term.append(outs[i])
                            else:                               # token type 1 --> second sentence
                                text.append(outs[i])

                    term = torch.stack([elem for elem in term], 0)  # size: [term_len x 768]
                    text = torch.stack([elem for elem in text], 0)  # size: [text_len x 768]
                    
                    if self.pooling == 'concat-mean':
                        term = torch.mean(term, dim=0)                  # size: [768]
                        text = torch.mean(text, dim=0)                  # size: [768]
                    elif self.pooling == 'concat-max':
                        term = torch.max(term, dim=0)[0]      if term.size()[0] > 1   else    term.squeeze()
                        text = torch.max(text, dim=0)[0]      if text.size()[0] > 1   else    text.squeeze()
                    else:
                        raise NotImplementedError

                    aux.append(torch.cat((term, text)))             # size: [1536]
                
                else:
                    tmp = []
                    for i in range(seq_len):
                        if inps[i] == 102 or inps[i] == 101:    # skip CLS and SEP token
                            continue
                        if inps[i] == 0:                        # exit on first encounter of PAD token
                            break
                        if mask[i] == 1:                        # not masked --> not padding
                            tmp.append(outs[i])

                    tmp = torch.stack([elem for elem in tmp], 0)
                
                    if self.pooling == 'reduce-mean':
                        tmp = torch.mean(tmp, dim=0)
                    elif self.pooling == 'reduce-max':
                        tmp = torch.max(tmp, dim=0)[0]
                    else:
                        raise NotImplementedError
                    
                    aux.append(tmp)             # size: [1536]

                
            pooler_out = torch.stack([elem for elem in aux], 0) # size: [32 x 1536]

        else:
            # standard case - just take the CLS output
            pooler_out = bert_out['pooler_output']
        
        return pooler_out

class BertForSequenceClassificationWithCustomPooling(BertPreTrainedModel):
    """
    BERT model for classification.\n
    This module is composed of the BERT model with a custom pooler and the default classifier
    """
    def __init__(self, config, num_labels=5, pooling='concat-mean', pool_layer=-1, device='cuda'):
        super(BertForSequenceClassificationWithCustomPooling, self).__init__(config)
        self.bert       = BertModel(config)
        self.dropout    = torch.nn.Dropout(config.hidden_dropout_prob)
        
        self.pooler     = CustomPooling(pooling=pooling, layer=pool_layer, device=device)
        self.pool_layer = pool_layer
        self.pooling    = pooling

        hidden_size = config.hidden_size
        if 'concat' in pooling:
            hidden_size = 2 * hidden_size
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        self.loss_fn    = torch.nn.CrossEntropyLoss()
        #self.init_weights()


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        bert_out   = self.bert(input_ids, token_type_ids, attention_mask, output_hidden_states=True)        
        pooler_out = self.pooler(bert_out, input_ids, attention_mask, token_type_ids)
        pooler_out = self.dropout(pooler_out)
        logits     = self.classifier(pooler_out)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(loss=loss,
                                        logits=logits,
                                        attentions=None,
                                        hidden_states=None)


    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


# MultiLabel base class for Task C
## from https://utter.works/multi-label-text-classification-using-bert-the-mighty-transformer/
class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """
    BERT model for multi-label classification, so it uses a BinaryCrossEntropyLoss instead of the CategoricalCrossEntropyLoss\n
    In this way, each label has its own probability of being chosen (independent of the others)
    """
    def __init__(self, config, num_labels=5):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels

        self.bert       = BertModel(config)
        self.loss_fn    = torch.nn.BCEWithLogitsLoss()
        self.dropout    = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        #self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        bert_out   = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_out = self.dropout(bert_out['pooler_output'])
        logits     = self.classifier(pooled_out)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return SequenceClassifierOutput(loss=loss,
                                        logits=logits,
                                        attentions=None,
                                        hidden_states=None)

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

class RobertaForMultiLabelSequenceClassification(RobertaPreTrainedModel):
    """
    ROBERTA model for multi-label classification, so it uses a BinaryCrossEntropyLoss instead of the CategoricalCrossEntropyLoss\n
    In this way, each label has its own probability of being chosen (independent of the others)
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_labels=5):
        super(RobertaForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels

        self.roberta    = RobertaModel(config, add_pooling_layer=False)
        self.loss_fn    = torch.nn.BCEWithLogitsLoss()
        self.classifier = RobertaClassificationHead(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        bert_out   = self.roberta(input_ids, attention_mask=attention_mask)
        logits     = self.classifier(bert_out[0])

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return SequenceClassifierOutput(loss=loss,
                                        logits=logits,
                                        attentions=None,
                                        hidden_states=None)

    def freeze_bert_encoder(self):
        for param in self.roberta.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.roberta.parameters():
            param.requires_grad = True



""" Debug stuff """


def rename_keys_in_state_dict(path):
    if 'CORRECT' in path:   # already modified
        return
    new_path = path.replace('.ckpt', '_CORRECT.ckpt')
    checkpoint = torch.load(path)
    new_checkpoint = OrderedDict()

    for k, v in checkpoint.items():
        if k == 'state_dict':
            new_checkpoint[k] = OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                new_key = key.replace('encoder', 'model.bert', 1)
                new_checkpoint['state_dict'][new_key] = value
                print(key, new_key)
        else:
            new_checkpoint[k] = v

    torch.save(new_checkpoint, new_path)

def print_elems(data, limit=10):
    i = 0
    for elem in data:
        for key in elem.keys():
            print(f">> '{key}'\t:\t{elem[key] if key != 'input' else elem[key].keys()}") 
        print()
        i += 1
        if i == limit:
            break

def count_occurrences(filepath):
    counter = Counter()
    with open(filepath, 'r') as f:
        data = json.load(f)
    for elem in data:
        if len(elem['targets']) > 0:
            for obj in elem['targets']:
                counter[obj[2]]+=1
        else:
            counter['absent'] += 1
    listed = counter.most_common()
    tot = 0
    for i in listed:
        tot += i[1]
    for i in listed:
        print(f'{i[0]}  \t({i[1]})\tis {i[1]*100/tot:.2f}% of the total ({tot})')

