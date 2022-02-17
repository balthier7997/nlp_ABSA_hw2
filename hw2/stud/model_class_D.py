import torch
import pytorch_lightning as pl

try:
    from utils               import *
    from dataset_class_extra import CategorySentimentDatasetBERT
except:
    from stud.utils               import *
    from stud.dataset_class_extra import CategorySentimentDatasetBERT

from tqdm   import tqdm
from pprint import pprint
from typing       import List, Dict

from torch.cuda      import FloatTensor
from transformers    import BertForSequenceClassification, RobertaForSequenceClassification, AdamW


# model using BertForSequenceClassification for task B   (has no nn.Module class, only pl.LightningModule)
class CategorySentimentModelBERT(pl.LightningModule):
    def __init__(self, hparams):
        super(CategorySentimentModelBERT, self).__init__()
        self.save_hyperparameters(hparams)
        if 'roberta' not in hparams['bert_type']:
            self.model = BertForSequenceClassification.from_pretrained(hparams['bert_type'],
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
        loss    = outputs.loss
        preds   = torch.argmax(outputs.logits, dim=-1)
        macro_f1, micro_f1 = self.compute_f1_score(batched_preds=preds, batch=batch, display=False)
        self.log('train_macro_f1', macro_f1, prog_bar=True,  on_epoch=True, on_step=False)
        self.log('train_micro_f1', micro_f1, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_loss',     loss,     prog_bar=True,  on_epoch=True, on_step=False)
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
        results = self.process_preds_d( batched_preds=batched_preds, 
                                        listed_texts=batch['texts'],
                                        listed_terms=batch['categories'],
                                        categories_list=batch['categories_list'], debug=False)
        macro_f1, micro_f1 = eval_task_D(batch['categories_list'], results, display=display)
        return FloatTensor([macro_f1], device=self.device), FloatTensor([micro_f1], device=self.device)


    def process_preds_d(self, batched_preds, listed_terms, listed_texts, categories_list, debug=False) -> List[Dict]:
        ø_itos = self.hparams['l_vocab_itos']
        
        result = []                # outer list
        þ = {'categories' : []}    # dict for categories
        current_text = ''
        for pred, term, text in zip(batched_preds, listed_terms, listed_texts):
            pred = pred.item()      # pred is a tensor
        
            if text != current_text:                            # meaning I changed the sentence
                if len(þ['categories']) > 0:                    # if I have some predictions pending
                    result.append(þ)                            # append them in the outer list
                þ = {'categories' : [[term, ø_itos[pred]]]}     # and reset and save the current pred

            else:
                þ['categories'].append([term, ø_itos[pred]]) # otherwise if the sentence is the same as before just append the next pred

            current_text = text
        
        # quando term == '' io resetto þ, ma in questo modo aggiungo due volte un array vuoto
        if len(þ['categories']) > 0:
            result.append(þ)

        if debug:
            print()
            print("batched preds      ", batched_preds)
            print("listed categories  ", listed_terms)
            print("listed texts       ", listed_texts)                      
            print("predicted polarity ", result[0]['categories'])
            print("gold category      ", categories_list[0]['categories'])
        assert len(result) == len(categories_list)
        #assert len(result[0]['categories']) == len(categories_list[0]['categories'])
        return result


    @torch.no_grad()
    def predict_task_d(self, raw_data: List[Dict], l_vocab, device='cpu'):
        print(">> Preprocessing data for task D")
        self.model.eval()
        device = self.hparams['device']
        data = CategorySentimentDatasetBERT.preprocessing_task_d(raw_data=raw_data, 
                                                            l_vocab=l_vocab, 
                                                            testing=True,
                                                            device=device)
        curr    = ''
        result  = [] 
        buffer  = []
        print(">> Predicting task D ...")
        progress_bar = tqdm(range(len(data)))
        for elem in data:
            if elem['text'] != curr:
                if len(buffer) > 0:
                    result.extend(self.sentence_predict_task_d(buffer))
                    buffer = []
            buffer.append(elem)
            curr = elem['text']                
            progress_bar.update(1)
        result.extend(self.sentence_predict_task_d(buffer))
        return result


    @torch.no_grad()
    def sentence_predict_task_d(self, buffer) -> List[Dict]:
        # in buffer I have all the couple <sentence, category> that that particular sentence has
        # create batch starting from the buffer
        batch = CategorySentimentDatasetBERT.collate_fn(batch=buffer,
                                                        testing=True,
                                                        first=self.hparams['first'],
                                                        tokenizer=self.hparams['tokenizer'],
                                                        roberta=('roberta' in self.hparams['bert_type']))
        test_batch = {k: v.to(self.device) for k, v in batch.items() if k not in self.blacklist}
        
        outputs = self.model(**test_batch)
        preds   = torch.argmax(outputs.logits, dim=-1)
        return self.process_preds_d(batched_preds=preds,
                                    listed_terms=batch['categories'],
                                    listed_texts=batch['texts'],
                                    categories_list=batch['categories_list'],
                                    debug=False)


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams['learning_rate'])



class CategorySentimentModelROBERTA(CategorySentimentModelBERT):
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
        self.blacklist = ['categories', 'texts', 'categories_list']
        print(self.model.classifier)
