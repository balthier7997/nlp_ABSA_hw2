from logging import raiseExceptions
import torch
import pytorch_lightning as pl

from tqdm           import tqdm
from pprint         import pprint
from typing         import List, Dict
from torch.cuda     import FloatTensor
from transformers   import BertForSequenceClassification, AdamW, RobertaForSequenceClassification

try:
    from utils         import *
    from dataset_class import TermPolarityDatasetBERT, TermPolarityDatasetBERT_noDummy
except:
    from stud.utils         import *
    from stud.dataset_class import TermPolarityDatasetBERT, TermPolarityDatasetBERT_noDummy

# model using BertForSequenceClassification for task B   (has no nn.Module class, only pl.LightningModule)
class TermPolarityModelBERT(pl.LightningModule):
    def __init__(self, hparams):
        super(TermPolarityModelBERT, self).__init__()
        self.save_hyperparameters(hparams)
        if 'roberta' not in hparams['bert_type']:
            if hparams['custom-pooler'] is not None:
                self.model = BertForSequenceClassificationWithCustomPooling.from_pretrained(hparams['bert_type'], 
                                                                                            num_labels=hparams['num_classes'],
                                                                                            pooling=hparams['custom-pooler'],
                                                                                            pool_layer=hparams['pooler-layer'], 
                                                                                            device=hparams['device'])
            else:
                self.model = BertForSequenceClassification.from_pretrained(hparams['bert_type'], num_labels=hparams['num_classes'])

            if hparams['classifier'] != 'base':
                self.model.classifier = build_custom_cls_head(  hidden_dim=hparams['hidden_dim'],
                                                                activation=hparams['classifier'],
                                                                num_layers=hparams['layers'],
                                                                num_labels=hparams['num_classes'],
                                                                dropout_pr=hparams['dropout'])
        self.blacklist = ['terms', 'texts', 'targets_list']
        
  
    def training_step(self, batch, batch_nb):
        # I need only part of the batch to train
        train_batch = {k: v.to(self.device) for k, v in batch.items() if k if k not in self.blacklist}
        outputs = self.model(**train_batch)
        loss    = outputs.loss
        preds   = torch.argmax(outputs.logits, dim=-1)
        macro_f1, micro_f1 = self.compute_f1_score(batched_preds=preds, batch=batch, display=False)
        self.log('train_macro_f1', macro_f1, prog_bar=True,  on_epoch=True, on_step=False)
        self.log('train_micro_f1', micro_f1, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        outputs.loss.backward(retain_graph=True)
        return loss


    def validation_step(self, batch, batch_nb):
        # I need only part of the batch to train
        valid_batch = {k: v.to(self.device) for k, v in batch.items() if k not in self.blacklist}
        outputs = self.model(**valid_batch)
        preds   = torch.argmax(outputs.logits, dim=-1)
        macro_f1, micro_f1 = self.compute_f1_score(batched_preds=preds, batch=batch, display=True)
        self.log('macro_f1', macro_f1, prog_bar=True)
        self.log('micro_f1', micro_f1, prog_bar=False)
        self.log('val_loss', outputs.loss, prog_bar=False, on_epoch=True, on_step=False)
        return macro_f1, micro_f1


    def compute_f1_score(self, batched_preds, batch, display=False):
        results = self.process_preds_b(batched_preds=batched_preds, 
                                            listed_terms=batch['terms'],
                                            listed_texts=batch['texts'],
                                            targets_list=batch['targets_list'], debug=False)
        self.sanitize_labels(batch['targets_list'], absent_token=self.hparams['absent_token'])
        macro_f1, micro_f1 = eval_task_B(batch['targets_list'], results, display=False)
        return FloatTensor([macro_f1], device=self.device), FloatTensor([micro_f1], device=self.device)


    def process_preds_b(self, batched_preds, listed_terms, listed_texts, targets_list, debug=False) -> List[Dict]:
        ø_itos = self.hparams['l_vocab_itos']

        result = []             # outer list
        þ = {'targets' : []}    # dict for targets
        current_text = ''
        for pred, term, text in zip(batched_preds, listed_terms, listed_texts):
            pred = pred.item()      # pred is a tensor

            if term == self.hparams['absent_token']:                          # ABSENT class doesn't count
                if len(þ['targets']) > 0:
                    result.append(þ)                # so I'll just skip these predictions
                result.append({'targets' : []})     # and put an empty list
                þ = {'targets' : []}

            else:
                if text != current_text:                            # meaning I changed the sentence
                    if len(þ['targets']) > 0:                       # if I have some predictions pending
                        result.append(þ)                            # append it in the outer list
                    þ = {'targets' :   [[term, ø_itos[pred]]]}      # and reset and save the current pred

                else:
                    þ['targets'].append([term, ø_itos[pred]]) # otherwise if the sentence is the same as before just append the next pred

            current_text = text

        # quando term == '' io resetto þ, ma in questo modo aggiungo due volte un array vuoto
        if len(þ['targets']) > 0:
            result.append(þ)

        print("\nPREDS\n", result)          if debug else 0
        print("GOLDS\n", targets_list)      if debug else 0
        assert len(result) == len(targets_list)
        return result


    @torch.no_grad()
    def predict_task_b(self, raw_data: List[Dict], l_vocab, device='cpu'):
        print(">> Preprocessing task B ...")
        self.model.eval()
        data = TermPolarityDatasetBERT.preprocessing_task_b(raw_data=raw_data, 
                                                            l_vocab=l_vocab, 
                                                            testing=True,
                                                            device=device,
                                                            absent_token=self.hparams['absent_token'])
        curr    = ''
        result  = [] 
        buffer  = []
        print(">> Predicting task B ...")
        progress_bar = tqdm(range(len(data)))
        for elem in data:
            if elem['text'] != curr:
                if len(buffer) > 0:
                    result.extend(self.sentence_predict_task_b(buffer))
                    buffer = []
            buffer.append(elem)
            curr = elem['text']                
            progress_bar.update(1)
        result.extend(self.sentence_predict_task_b(buffer))
        return result


    @torch.no_grad()
    def sentence_predict_task_b(self, buffer) -> List[Dict]:
        # in buffer I have all the couple <sentence, aspect term> that that particular sentence has
        # create batch starting from the buffer
        batch = TermPolarityDatasetBERT.test_collate_fn(batch=buffer,
                                                        tokenizer=self.hparams['tokenizer'],
                                                        roberta=('roberta' in self.hparams['bert_type']),
                                                        first=self.hparams['first'])
        test_batch = {k: v.to(self.device) for k, v in batch.items() if k not in self.blacklist}
        
        outputs = self.model(**test_batch)
        preds   = torch.argmax(outputs.logits, dim=-1)
        #preds   = torch.argmax(outputs.logits[:,:4], dim=-1)       # preds without the absent class
        return self.process_preds_b(preds,
                                    listed_terms=batch['terms'],
                                    listed_texts=batch['texts'],
                                    targets_list=batch['targets_list'], debug=False)
    

    @classmethod
    def sanitize_labels(self, labels, absent_token):
        """
        I add dummy labels in order to split the sentences and train
        When I evaluate I need to remove them to respect the output requirement
        """
        for label in labels:
            if label['targets'][0][1] == absent_token:
                label['targets'] = []


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams['learning_rate'])


#Old Class
class TermPolarityModelBERT_noDummy(pl.LightningModule):
    def __init__(self, hparams):
        super(TermPolarityModelBERT_noDummy, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = BertForSequenceClassification.from_pretrained(hparams['bert_type'],
                                                                num_labels=hparams['num_classes'])
        
        if hparams['classifier'] != 'base':
            self.model.classifier = build_custom_cls_head(  hidden_dim=hparams['hidden_dim'],
                                                            activation=hparams['classifier'],
                                                            num_layers=hparams['layers'],
                                                            num_labels=hparams['num_classes'],
                                                            dropout_pr=hparams['dropout'])
        self.blacklist = ['terms', 'texts', 'targets_list']
        
  
    def training_step(self, batch, batch_nb):
        # I need only part of the batch to train
        train_batch = {k: v.to(self.device) for k, v in batch.items() if k if k not in self.blacklist}
        outputs = self.model(**train_batch)
        loss    = outputs.loss
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        outputs.loss.backward(retain_graph=True)
        return loss


    def validation_step(self, batch, batch_nb):
        # I need only part of the batch to train
        valid_batch = {k: v.to(self.device) for k, v in batch.items() if k not in self.blacklist}
        outputs = self.model(**valid_batch)
        preds   = torch.argmax(outputs.logits, dim=-1)
        
        macro_f1, micro_f1 = self.compute_f1_score(batched_preds=preds, batch=batch, display=True)
        self.log('macro_f1', macro_f1, prog_bar=True)
        self.log('micro_f1', micro_f1, prog_bar=False)
        return macro_f1, micro_f1


    def compute_f1_score(self, batched_preds, batch, display=False):
        results = self.process_preds_b(batched_preds=batched_preds, 
                                            listed_terms=batch['terms'],
                                            listed_texts=batch['texts'],
                                            targets_list=batch['targets_list'], debug=False)
        self.sanitize_labels(batch['targets_list'])
        macro_f1, micro_f1 = eval_task_B(batch['targets_list'], results, display=display)
        return FloatTensor([macro_f1], device=self.device), FloatTensor([micro_f1], device=self.device)


    def process_preds_b(self, batched_preds, listed_terms, listed_texts, targets_list, debug=False) -> List[Dict]:        
        ø_itos = self.hparams['l_vocab_itos']

        result = []             # outer list
        þ = {'targets' : []}    # dict for targets
        current_text = ''
        for pred, term, text in zip(batched_preds, listed_terms, listed_texts):
            print(f"\n<{term}>")                    if debug else 0

            pred = pred.item()      # pred is a tensor

            if text != current_text:                            # meaning I changed the sentence
                if len(þ['targets']) > 0:                       # if I have some predictions pending
                    result.append(þ)                            # append it in the outer list
                þ = {'targets' :   [[term, ø_itos[pred]]]}      # and reset and save the current pred

            else:
                þ['targets'].append([term, ø_itos[pred]]) # otherwise if the sentence is the same as before just append the next pred

            current_text = text
            print(f">> þ is {þ}")           if debug else 0

        # quando term == '' io resetto þ, ma in questo modo aggiungo due volte un array vuoto
        if len(þ['targets']) > 0:
            result.append(þ)

        print("\nPREDS\n", result)          if debug else 0
        print("GOLDS\n", targets_list)      if debug else 0
        assert len(result) == len(targets_list)
        return result


    @torch.no_grad()
    def predict_task_b(self, raw_data: List[Dict], l_vocab, device='cpu'):
        print(">> Preprocessing. . .")
        self.model.eval()
        #device = self.hparams['device']

        # in data ho le frasi senza termini
        data = TermPolarityDatasetBERT_noDummy.test_preprocessing_task_b(raw_data=raw_data, 
                                                            l_vocab=l_vocab, 
                                                            testing=True,
                                                            device=device)
        curr    = ''
        result  = [] 
        buffer  = []
        count   = 0
        print(">> Predicting task B ...")
        progress_bar = tqdm(range(len(data)))
        for elem in data:
            if elem['text'] != curr:
                if len(buffer) > 0:
                    result.extend(self.sentence_predict_task_b(buffer))
                    buffer = []

            if elem['targets'][0][1] == '':
                elem['targets'] = []
                result.extend([{'targets' : []}])
                count += 1
            else:
                buffer.append(elem)
                curr = elem['text']                
            progress_bar.update(1)
        result.extend(self.sentence_predict_task_b(buffer))
        print(f">>> Found {count} absent flags")
        return result


    @torch.no_grad()
    def sentence_predict_task_b(self, buffer) -> List[Dict]:
        # in buffer I have all the couple <sentence, aspect term> that that particular sentence has
        # create batch starting from the buffer
        batch = TermPolarityDatasetBERT_noDummy.test_collate_fn(buffer, self.hparams['tokenizer'], first=self.hparams['first'])
        test_batch = {k: v.to(self.device) for k, v in batch.items() if k not in self.blacklist}
        
        outputs = self.model(**test_batch)
        preds   = torch.argmax(outputs.logits, dim=-1)
        return self.process_preds_b(preds,
                                    listed_terms=batch['terms'],
                                    listed_texts=batch['texts'],
                                    targets_list=batch['targets_list'], debug=False)
    

    @classmethod
    def sanitize_labels(self, labels):
        """
        I add dummy labels in order to split the sentences and train
        When I evaluate I need to remove them to respect the output requirement
        """
        for label in labels:
            if label['targets'][0][1] == '':
                label['targets'] = []


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams['learning_rate'])



class TermPolarityModelROBERTA(TermPolarityModelBERT):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.save_hyperparameters(hparams)
        
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=hparams['num_classes'])
        if hparams['classifier'] != 'base':
            self.model.classifier = CustomRobertaClassificationHead(hidden_dim=hparams['hidden_dim'],
                                                                    activation=hparams['classifier'],
                                                                    num_layers=hparams['layers'],
                                                                    num_labels=hparams['num_classes'],
                                                                    dropout_pr=hparams['dropout'])
        self.blacklist = ['terms', 'texts', 'targets_list']
        print(self.model.classifier)
