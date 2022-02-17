import torch
import pytorch_lightning as pl

from tqdm   import tqdm
from pprint import pprint
from typing import List, Dict
from torch.cuda    import FloatTensor
from transformers  import AdamW

try:
    from utils  import *
    from dataset_class_extra import CategoryExtractionDatasetBERT
except:
    from stud.utils  import *
    from stud.dataset_class_extra import CategoryExtractionDatasetBERT


class CategoryExtractionMultiLabelModelBERT(pl.LightningModule):
    def __init__(self, hparams):
        super(CategoryExtractionMultiLabelModelBERT, self).__init__()
        self.save_hyperparameters(hparams)
        if 'roberta' not in hparams['bert_type']:
            self.model = BertForMultiLabelSequenceClassification.from_pretrained(hparams['bert_type'],
                                                                    num_labels=hparams['num_classes'])
                    
            if hparams['classifier'] != 'base':
                self.model.classifier = build_custom_cls_head(  hidden_dim=hparams['hidden_dim'],
                                                                activation=hparams['classifier'],
                                                                num_layers=hparams['layers'],
                                                                num_labels=hparams['num_classes'],
                                                                dropout_pr=hparams['dropout'])
        self.blacklist = ['categories', 'texts', 'categories_list']
        
  
    def training_step(self, batch, batch_nb):
        # I need only part of the batch to train
        train_batch = {k: v.to(self.device) for k, v in batch.items() if k if k not in self.blacklist}
        outputs = self.model(**train_batch)
        loss = outputs.loss
        preds   = torch.sigmoid(outputs.logits)
        macro_f1, micro_f1 = self.compute_f1_score(preds=preds, batch=batch, display=False)
        self.log('train_macro_f1', macro_f1, prog_bar=True,  on_epoch=True, on_step=False)
        self.log('train_micro_f1', micro_f1, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        loss.backward(retain_graph=True)
        return loss


    def validation_step(self, batch, batch_nb):
        # I need only part of the batch to train
        valid_batch = {k: v.to(self.device) for k, v in batch.items() if k not in self.blacklist}
        outputs = self.model(**valid_batch)
        logits  = outputs.logits 
        preds   = torch.sigmoid(logits)
        macro_f1, micro_f1 = self.compute_f1_score(preds=preds, batch=batch, display=False)
        self.log('macro_f1', macro_f1, prog_bar=True)
        self.log('micro_f1', micro_f1, prog_bar=False)
        self.log('val_loss', outputs.loss, prog_bar=False, on_epoch=True, on_step=False)
        return macro_f1, micro_f1


    def compute_f1_score(self, preds, batch, display=False):
        results = self.process_preds_c(preds=preds)
        macro_f1, micro_f1 = eval_task_C(batch['categories_list'], results, display=display)
        return FloatTensor([macro_f1], device=self.device), FloatTensor([micro_f1], device=self.device)


    def process_preds_c(self, preds) -> List[Dict]:
        """ 
        Processes the numerical output of the model in order to have a List of Dictionaries
        """
        THRESHOLD = 0.5
        ø_itos = self.hparams['l_vocab_itos']
        result = []
        þ = {'categories' : []}
        for pred in preds:
            out = [0, 0, 0, 0, 0]
            
            for i in range(len(pred)):
                if pred[i].item() > THRESHOLD:
                    out[i] = 1
                    þ['categories'].append([ø_itos[i]])
            # if no logits is above the thresholds, then I take the maximum
            # this is to ensure that there's at least one output category for each data sample
            if 1 not in out:
                i = torch.argmax(pred).item()
                out[i] = 1
                þ['categories'].append([ø_itos[i]])

            result.append(þ)
            þ = {'categories' : []}
        return result

    
    @torch.no_grad()
    def predict_task_c(self, raw_data: List[Dict], l_vocab, device='cpu'):
        print(">> Preprocessing task C ...")
        self.model.eval()
        device = self.hparams['device']
        data = CategoryExtractionDatasetBERT.preprocessing_task_c(raw_data=raw_data, l_vocab=l_vocab, testing=True, device=device)
        curr = ''
        result  = [] 
        buffer  = []
        print(">> Predicting task C ...")
        progress_bar = tqdm(range(len(data)))
        for elem in data:
            if elem['text'] != curr:
                if len(buffer) > 0:
                    result.extend(self.sentence_predict_task_c(buffer))
                    buffer = []
            buffer.append(elem)
            curr = elem['text']                
            progress_bar.update(1)
        result.extend(self.sentence_predict_task_c(buffer))
        return result


    @torch.no_grad()
    def sentence_predict_task_c(self, buffer) -> List[Dict]:
        batch = CategoryExtractionDatasetBERT.collate_fn(batch=buffer,
                                                        tokenizer=self.hparams['tokenizer'],
                                                        roberta=('roberta' in self.hparams['bert_type']),
                                                        testing=True)
        test_batch = {k: v.to(self.device) for k, v in batch.items() if k not in self.blacklist}
        outputs = self.model(**test_batch)
        preds  = torch.sigmoid(outputs.logits)
        return self.process_preds_c(preds=preds)


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams['learning_rate']) ## default: 5e-5, recommended on paper: 3e-5


class CategoryExtractionMultiLabelModelROBERTA(CategoryExtractionMultiLabelModelBERT):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.save_hyperparameters(hparams)
        self.model = RobertaForMultiLabelSequenceClassification.from_pretrained(hparams['bert_type'],
                                                                num_labels=hparams['num_classes'])
                
        if hparams['classifier'] != 'base':
            self.model.classifier = CustomRobertaClassificationHead(hidden_dim=hparams['hidden_dim'],
                                                                    activation=hparams['classifier'],
                                                                    num_layers=hparams['layers'],
                                                                    num_labels=hparams['num_classes'],
                                                                    dropout_pr=hparams['dropout'])
        self.blacklist = ['categories', 'texts', 'categories_list']
        print(self.model.classifier)



''' Old class not using the multi label base class -- poorer results
class CategoryExtractionModelBERT(pl.LightningModule):
    def __init__(self, hparams):
        super(CategoryExtractionModelBERT, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = AutoModelForSequenceClassification.from_pretrained(hparams['bert_type'],
                                                                num_labels=hparams['num_classes'])
                
        if hparams['classifier'] != 'base':
            self.model.classifier = build_custom_cls_head(  hidden_dim=hparams['hidden_dim'],
                                                            activation=hparams['classifier'],
                                                            num_layers=hparams['layers'],
                                                            num_labels=hparams['num_classes'],
                                                            dropout_pr=hparams['dropout'])
        self.blacklist = ['categories', 'texts', 'categories_list']
        
  
    def training_step(self, batch, batch_nb):
        # I need only part of the batch to train
        train_batch = {k: v.to(self.device) for k, v in batch.items() if k if k not in self.blacklist}
        outputs = self.model(**train_batch)
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        loss.backward(retain_graph=True)
        return loss


    def validation_step(self, batch, batch_nb):
        # I need only part of the batch to train
        valid_batch = {k: v.to(self.device) for k, v in batch.items() if k not in self.blacklist}
        outputs = self.model(**valid_batch)
        preds   = outputs.logits 
        preds   = torch.softmax(preds, dim=-1)
        macro_f1, micro_f1 = self.compute_f1_score(preds=preds, batch=batch, display=False)
        self.log('macro_f1', macro_f1, prog_bar=True)
        self.log('micro_f1', micro_f1, prog_bar=False)
        return macro_f1, micro_f1


    def compute_f1_score(self, preds, batch, display=False):
        results = self.process_preds_c(preds=preds)
        macro_f1, micro_f1 = eval_task_C(batch['categories_list'], results, display=display)
        return FloatTensor([macro_f1], device=self.device), FloatTensor([micro_f1], device=self.device)


    def process_preds_c(self, preds) -> List[Dict]:
        """
        Processes the numerical output of the model in order to have a List of Dictionaries
        """
        THRESHOLD = 0.2
        ø_itos = self.hparams['l_vocab_itos']
        result = []
        þ = {'categories' : []}
        for pred in preds:
            out = [0, 0, 0, 0, 0]
            for i in range(len(pred)):
                if pred[i].item() > THRESHOLD:# or ø_itos[i] in texts:
                    out[i] = 1
                    þ['categories'].append([ø_itos[i]])
            if 1 not in out:            # se nessuno dei logits supera la threshold allora per evitare problemi ritorna solo il massimo
                i = torch.argmax(pred).item()
                out[i] = 1
                þ['categories'].append([ø_itos[i]])
            result.append(þ)
            þ = {'categories' : []}
        return result

    
    @torch.no_grad()
    def predict_task_c(self, raw_data: List[Dict], l_vocab, device='cpu'):
        print(">> Preprocessing. . .")
        self.model.eval()
        device = self.hparams['device']
        data = CategoryExtractionDatasetBERT.preprocessing_task_c(raw_data=raw_data, l_vocab=l_vocab, testing=True, device=device)
        curr = ''
        result = [] 
        buffer  = []
        print(">> Predicting. . .")
        progress_bar = tqdm(range(len(data)))
        for elem in data:
            if elem['text'] != curr:
                if len(buffer) > 0:
                    result.extend(self.sentence_predict_task_c(buffer))
                    buffer = []
            buffer.append(elem)
            curr = elem['text']                
            progress_bar.update(1)
        return result


    @torch.no_grad()
    def sentence_predict_task_c(self, buffer) -> List[Dict]:
        batch = CategoryExtractionDatasetBERT.collate_fn(buffer, self.hparams['tokenizer'])
        test_batch = {k: v.to(self.device) for k, v in batch.items() if k not in self.blacklist}
        outputs = self.model(**test_batch)
        preds   = outputs.logits
        preds   = torch.softmax(preds, dim=-1)
        return self.process_preds_c(preds=preds)


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams['learning_rate'])
'''