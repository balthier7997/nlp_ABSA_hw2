import re
import torch
import numpy as np
import torch.optim as optim
import pytorch_lightning as pl

from torch  import nn
from tqdm   import tqdm
from pprint import pprint
from torch.tensor import Tensor
from typing       import List, Tuple, Dict

from torch.cuda      import FloatTensor
from sklearn.metrics import f1_score
from transformers    import AutoModelForTokenClassification, AdamW, RobertaForTokenClassification

try:
    from utils         import *
    from dataset_class import TermIdentificationDataset
except:
    from stud.utils         import *
    from stud.dataset_class import TermIdentificationDataset

# needed only to perform some test with the base model without the lightning stuff
class HParams():
    def __init__(self, vocab,
                embedding_dim=100,
                lstm_hidden_dim=128,
                lstm_bidirectional=False,
                lstm_layers=1,
                num_classes=3,
                dropout=0.0,
                device='cuda'):

        self.device      = device
        self.vocab       = vocab
        self.vocab_size  = len(vocab)
        
        self.embedding_dim = embedding_dim

        self.lstm_hidden_dim    = lstm_hidden_dim
        self.lstm_bidirect = lstm_bidirectional
        self.lstm_layers        = lstm_layers
        
        self.num_classes = num_classes
        self.dropout     = dropout


# class to manage the embeddings
class PreTrainedEmbeddingLayer():
    def __init__(self, hparams):
        self.vocab          = hparams['vocab']
        self.vocab_size     = hparams['vocab_size']
        self.embedding_dim  = hparams['embedding_dim']
        self.emb_type       = hparams['embedding_type']
        self.device         = hparams['device']
        
        if self.emb_type   == 'glove':
            embedding_path   = f"hw2/stud/glove/glove.{hparams['B']}B.{self.embedding_dim}d.txt"
            embedding_weights = self.load_pretrained_embeddings(embedding_path)
            self.embedding    = nn.Embedding.from_pretrained(embedding_weights)

        elif self.emb_type == 'domain-restaurant':
            embedding_path    = "hw2/stud/domain_embedding/restaurant_emb.vec"
            self.embedding_dim = hparams['embedding_dim'] = 100
            embedding_weights  = self.load_pretrained_embeddings(embedding_path)
            self.embedding     = nn.Embedding.from_pretrained(embedding_weights)
        
        elif self.emb_type == 'domain-laptop':
            embedding_path = "hw2/stud/domain_embedding/laptop_emb.vec"
            self.embedding_dim = hparams['embedding_dim'] = 100
            embedding_weights  = self.load_pretrained_embeddings(embedding_path)
            self.embedding     = nn.Embedding.from_pretrained(embedding_weights)
        
        elif self.emb_type == 'glove+rest_domain':
            # first load glove embeddings
            embedding_path    = f"hw2/stud/glove/glove.{hparams['B']}B.{self.embedding_dim}d.txt"
            embedding_weights = self.load_pretrained_embeddings(embedding_path)
            
            # then concatenate the domain specific embeddings
            self.embedding_dim       += 100 
            hparams['embedding_dim'] += 100
            embedding_path    = "hw2/stud/domain_embedding/restaurant_emb.vec"
            embedding_weights = self.concatenate_embeddings(embedding_path, embedding_weights)
            self.embedding    = nn.Embedding.from_pretrained(embedding_weights)
        
        elif self.emb_type == 'glove+lapt_domain':
            # first load glove embeddings
            embedding_path    = f"hw2/stud/glove/glove.{hparams['B']}B.{self.embedding_dim}d.txt"
            embedding_weights = self.load_pretrained_embeddings(embedding_path)
            
            # then concatenate the domain specific embeddings
            self.embedding_dim       += 100 
            hparams['embedding_dim'] += 100
            embedding_path    = "hw2/stud/domain_embedding/laptop_emb.vec"
            embedding_weights = self.concatenate_embeddings(embedding_path, embedding_weights)
            self.embedding    = nn.Embedding.from_pretrained(embedding_weights)

        elif self.emb_type == 'glove+all_domains':
            # first load glove embeddings
            embedding_path    = f"hw2/stud/glove/glove.{hparams['B']}B.{self.embedding_dim}d.txt"
            embedding_weights = self.load_pretrained_embeddings(embedding_path)
            
            # then concatenate the domain specific embeddings
            self.embedding_dim       += 100 
            hparams['embedding_dim'] += 100
            embedding_weights = self.combine_n_concatenate_embeddings(embedding_path, embedding_weights)
            self.embedding    = nn.Embedding.from_pretrained(embedding_weights)

        elif self.emb_type == 'all_domains':
            # only use specific domains
            self.embedding_dim       = 100 
            hparams['embedding_dim'] = 100
            embedding_weights = self.combine_embeddings()
            self.embedding    = nn.Embedding.from_pretrained(embedding_weights)



    def load_emb(self, embedding_path):
        # I load the embeddings
        mapping = {}
        print(f">> Loading embeddings from '{embedding_path}'")

        num_lines = sum(1 for _ in open(embedding_path,'r'))   # only required for the nice progbar
        with open(embedding_path) as f:
            for line in tqdm(f, total=num_lines):
                splitted = line.split()
                try:    # skip problematic lines in glove.840B.300d
                    float(splitted[1])
                except ValueError as e:
                    print(splitted[0], splitted[1])
                    continue
                mapping[splitted[0]] = FloatTensor([float(i) for i in splitted[1:]], device=self.device)
        print(f">> Loaded {len(mapping)} embeddings.")
        return mapping


    def load_pretrained_embeddings(self, embedding_path):
        mapping = self.load_emb(embedding_path)
        weights = torch.zeros(self.vocab_size, self.embedding_dim, device=self.device)
        print(f">> Shape of weights matrix W: {weights.shape}")
        count = 0
        # save the embedding of all words in vocab
        # if a word doesn't have an embedding, I create a random vector
        print(">> Saving the embeddings...")
        for word in self.vocab.stoi.keys():
            if word.lower() in mapping:         #.lower() when I search in glove
                aux = mapping[word.lower()]  
            #if word in mapping:         
            #    aux = mapping[word]  
            else:
                aux = FloatTensor(np.random.normal(scale=0.6, size=(self.embedding_dim, )), device=self.device)
                count += 1
            weights[self.vocab[word]] = aux     # but when I save it, I save it with (possible) capital letters
                                                # in this way I may have 2+ copies of the same vector (The and the)
                                                # but since I need to match the *exact* words this is needed
        weights[self.vocab["<pad>"]] = torch.zeros(size=(self.embedding_dim,), device=self.device)
        print(f">> {count} words in my vocab not present in glove (random vectors created)")
        return weights


    def concatenate_embeddings(self, embedding_path, embedding_weights):
        # loading the embeddings
        mapping = self.load_emb(embedding_path)
        weights = torch.zeros(self.vocab_size, self.embedding_dim, device=self.device)
        print(f">> Shape of weights matrix W: {weights.shape}")
        count = 0
        # concatenating the embeddings
        for word in self.vocab.stoi.keys():
            if word.lower() in mapping:         #.lower() when I search in glove
                aux = mapping[word.lower()]  
            else:
                aux = FloatTensor(np.random.normal(scale=0.6, size=(100, )), device=self.device) # size is always 100 for these embeddings
                count += 1
            concatenation = torch.cat((embedding_weights[self.vocab[word]], aux), dim=0)   
            weights[self.vocab[word]] = concatenation  

        weights[self.vocab["<pad>"]] = torch.zeros(size=(self.embedding_dim,), device=self.device)
        print(f">> {count} glove embeddings concatenated with random vectors")
        return weights


    def combine_embeddings(self):
        lapt_mapping = self.load_emb("hw2/stud/domain_embedding/laptop_emb.vec")
        rest_mapping = self.load_emb("hw2/stud/domain_embedding/restaurant_emb.vec")
        lapt_keys = set(lapt_mapping.keys())
        rest_keys = set(rest_mapping.keys())
        valid_lapt_keys = lapt_keys.difference(rest_keys)
        weights = torch.zeros(self.vocab_size, self.embedding_dim, device=self.device)
        print(f">> Shape of weights matrix W: {weights.shape}")

        count_missing = count_lapt_added = count_rest_added = 0
        for word in self.vocab.stoi.keys():
            if word.lower() in valid_lapt_keys:         #.lower() when I search in glove
                aux = lapt_mapping[word.lower()] 
                count_lapt_added += 1 
            elif word.lower() in rest_keys:
                aux = rest_mapping[word.lower()]
                count_rest_added += 1
            else:
                aux = FloatTensor(np.random.normal(scale=0.6, size=(100, )), device=self.device) # size is always 100 for these embeddings
                count_missing += 1
            weights[self.vocab[word]] = aux
        weights[self.vocab["<pad>"]] = torch.zeros(size=(self.embedding_dim,), device=self.device)
        print(f">> {count_missing} words in my vocab not present in the domain embeddings (random vectors created)")
        print(f">> {count_lapt_added} embeddings used from laptop domain")
        print(f">> {count_rest_added} embeddings used from restaurant domain")
        return weights


    def combine_n_concatenate_embeddings(self, embedding_path, embedding_weights):
        lapt_mapping = self.load_emb("hw2/stud/domain_embedding/laptop_emb.vec")
        rest_mapping = self.load_emb("hw2/stud/domain_embedding/restaurant_emb.vec")
        lapt_keys = set(lapt_mapping.keys())
        rest_keys = set(rest_mapping.keys())
        valid_lapt_keys = lapt_keys.difference(rest_keys)
        # I will concatenate all the restaurant embeddings but for the valid_lapt_keys words
        # which are ~11244 words relevant to laptop (way less then the words relevant for restaurants)
        # of these, only 2167 are actually in Glove
        count_missing = count_lapt_added = count_rest_added = 0
        weights = torch.zeros(self.vocab_size, self.embedding_dim, device=self.device)
        print(f">> Shape of weights matrix W: {weights.shape}")
        # concatenating the embeddings
        for word in self.vocab.stoi.keys():
            if word.lower() in valid_lapt_keys:         #.lower() when I search in glove
                aux = lapt_mapping[word.lower()] 
                count_lapt_added += 1 
            elif word.lower() in rest_keys:
                aux = rest_mapping[word.lower()]
                count_rest_added += 1
            else:
                aux = FloatTensor(np.random.normal(scale=0.6, size=(100, )), device=self.device) # size is always 100 for these embeddings
                count_missing += 1
            concatenation = torch.cat((embedding_weights[self.vocab[word]], aux), dim=0)   
            weights[self.vocab[word]] = concatenation  

        weights[self.vocab["<pad>"]] = torch.zeros(size=(self.embedding_dim,), device=self.device)
        print(f">> {count_missing} glove embeddings concatenated with random vectors")
        print(f">> {count_lapt_added} glove embeddings concatenated with laptops embedding")
        print(f">> {count_rest_added} glove embeddings concatenated with restaurants embedding")
        return weights


    def get_embeddings(self):
        return self.embedding


# faster model to test dimensions and stuff
class TermIdentificationBaseModule_FAST(nn.Module):
    def __init__(self, hparams, embedding):
        super(TermIdentificationBaseModule_FAST, self).__init__()
        self.embedding = embedding        
        self.lstm = nn.LSTM(hparams.embedding_dim,
                            hparams.lstm_hidden_dim, 
                            bidirectional=False,
                            num_layers=1)
        self.linear  = nn.Linear(hparams.lstm_hidden_dim, hparams.num_classes)

    def forward(self, x):
        embedded_x = self.embedding(x)
        lstm_out   = self.lstm(embedded_x)[0]
        return self.linear(lstm_out)  


# base model, used pre-trained embedding 
class TermIdentificationBaseModule(nn.Module):
    def __init__(self, hparams, embedding):
        super(TermIdentificationBaseModule, self).__init__()
        self.embedding = embedding
        
        self.dropout = nn.Dropout(hparams.dropout)
        
        self.lstm = nn.LSTM(hparams.embedding_dim, hparams.lstm_hidden_dim, 
                            bidirectional=hparams.lstm_bidirect,
                            num_layers=hparams.lstm_layers, 
                            dropout = hparams.dropout if hparams.lstm_layers > 1 else 0)

        self.lstm_output_dim = hparams.lstm_hidden_dim if hparams.lstm_bidirect is False else 2*hparams.lstm_hidden_dim

        self.linear1  = nn.Linear(self.lstm_output_dim,    self.lstm_output_dim//2)
        self.linear2  = nn.Linear(self.lstm_output_dim//2, self.lstm_output_dim//4)
        self.linear3  = nn.Linear(self.lstm_output_dim//4, self.lstm_output_dim//8)
        self.linear4  = nn.Linear(self.lstm_output_dim//8, hparams.num_classes)


    def forward(self, x):
        embedded_x = self.embedding(x)
        embedded_x = self.dropout(embedded_x)

        lstm_out = self.lstm(embedded_x)[0]
        lstm_out = self.dropout(lstm_out)
        
        logits = self.linear1(lstm_out)  
        logits = torch.relu(logits)      
        logits = self.linear2(logits)  
        logits = torch.relu(logits)
        logits = self.linear3(logits)
        logits = torch.relu(logits)
        logits = self.linear4(logits)     
        return logits


# pl module of the above classes
class TermIdentificationModel(pl.LightningModule):
    
    def __init__(self, hparams, embedding):
        super(TermIdentificationModel, self).__init__()
        self.save_hyperparameters(hparams)    
        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.hparams.l_vocab[self.hparams.pad_token])
        if self.hparams.model   == 'FAST':
            self.model = TermIdentificationBaseModule_FAST(self.hparams, embedding)
        elif self.hparams.model == 'BASE':
            self.model = TermIdentificationBaseModule(self.hparams, embedding)        
        
    def forward(self, x):
        logits = self.model(x)
        preds  = torch.argmax(logits, -1)
        return logits, preds


    def training_step(self, batch, batch_nb):
        inputs = batch[0]
        labels = batch[1]
        logits, _ = self.forward(inputs)
        
        # adapt the logits and labels to fit the format required for the loss function
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)

        loss = self.loss_function(logits, labels)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss                                     # always return the loss


    def validation_step(self, batch, batch_nb):
        inputs  = batch[0]
        labels  = batch[1]
        texts   = batch[2]
        targets = batch[3]

        _, preds = self.forward(inputs)
        
        # score computed on the Sequence Tagging problem
        F1_seq_tag = self.compute_f1_seqtag(preds, labels)
        self.log('macro_F1_seq_tag', F1_seq_tag, prog_bar=True)
        
        # score computed on the actual task
        F1_score = self.compute_f1_score(preds, texts, targets)
        self.log('F1_score', F1_score, prog_bar=True)

        return F1_score


    def compute_f1_seqtag(self, preds: Tensor, labels: Tensor, display: bool = False) -> FloatTensor:
        """
        Function to compute the macro F1 score on the Sequence Tagging problem (PoS, NER, ...) \n
        """
        preds    = preds.view( -1)
        labels   = labels.view(-1)
        indices  = labels != self.hparams.l_vocab['<pad>']
        macro_f1 = f1_score(labels[indices].tolist(),
                            preds[ indices].tolist(), 
                            average="macro",
                            zero_division=0)
        if display:
            micro_f1 = f1_score(labels[indices].tolist(), preds[indices].tolist(), average="micro", zero_division=0)
            class_f1 = f1_score(labels[indices].tolist(), preds[indices].tolist(), labels=range(len(self.hparams.l_vocab)), average=None, zero_division=0)
            print_seqtag_results(macro_f1, micro_f1, class_f1, self.hparams.l_vocab_itos)
        return FloatTensor([macro_f1], device=self.device)  # required in order to log it with cuda on 


    def process_preds_task_a(self, batched_preds: Tensor, batched_texts: Tensor) -> List[Dict]:
        """
        Take the output of the model and process it to be the output for task A                     \n
        Starting from the tagged sequence, it builds the set of terms that is the actual prediction
        """
        results = []
        þ = self.hparams.l_vocab_itos
        for predictions, text in zip(batched_preds, batched_texts):
            preds = []
            aux = ''
            for i in range(len(text)):
                ø = predictions[i].item()

                if þ[ø] == BEGINNING:
                    if aux != '':
                        preds.append(aux)
                    aux = text[i]

                elif þ[ø] == INSIDE:
                    if aux != '':      # it may happen that the model classifies the Beginning token as Inside
                        aux += ' '     # in this way it will not have a starting space and so it will match with the word (if it's the case)
                    aux += text[i]
                
                elif þ[ø] == OUTSIDE:
                    if aux != '':
                        preds.append(aux)
                        aux = ''

            # remove empty strings, if any
            while '' in preds:
                preds.remove('')

            results.append({'targets' : preds})
        return results


    def compute_f1_score(self, batched_preds: Tensor, batched_texts: Tensor, batched_targets: Tensor) -> FloatTensor:
        """
        Computes the F1 score on the actual task A      \n
        First it process the predictions in the batch   \n
        Then it evaluates them
        """
        results = self.process_preds_task_a(batched_preds, batched_texts)
        return FloatTensor([eval_task_A(batched_targets, results)], device=self.device)

    
    def predict_task_a(self, data: List[Dict]) -> List[Dict]:
        results = []
        max_len = 100
        for elem in data:
            # preprocess data
            words_vector = TermIdentificationDataset.tokenize_line(elem['text'])
            for _ in range(len(words_vector), max_len):
                words_vector.append(self.hparams.pad_token)
            idxs_vector  = TermIdentificationDataset.words2indexes(words_vector, self.hparams.pad_token, self.hparams.vocab, 'cpu')
            # actual predict
            batched_inputs = idxs_vector.unsqueeze(0).cpu()
            batched_texts  = [words_vector]
            res = self.single_predict_task_a(batched_inputs, batched_texts)
            results.append(res[0])
        return results


    @torch.no_grad()
    def single_predict_task_a(self, batched_inputs: Tensor, batched_texts: List[List]) -> List[Dict]:
        self.model.eval()
        _, batched_preds = self.forward(batched_inputs)
        # return a list with only one elem (a dict {'target' : set()})
        return self.process_preds_task_a(batched_preds, batched_texts)


    def configure_optimizers(self):
        return optim.Adam(self.parameters())


#########################################################################################################################################


# model using BERT  (has no nn.Module class, only pl.LightningModule)
class TermIdentificationModelBERT(pl.LightningModule):
    def __init__(self, hparams):
        super(TermIdentificationModelBERT, self).__init__()
        self.save_hyperparameters(hparams)
        if 'roberta' in hparams['bert_type']:
            self.model = RobertaForTokenClassification.from_pretrained( hparams['bert_type'],
                                                                        num_labels=hparams['num_classes'])
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(hparams['bert_type'],
                                                                        num_labels=hparams['num_classes'])

        if hparams['classifier'] != 'base':
            self.model.classifier = build_custom_cls_head(  hidden_dim=hparams['hidden_dim'],
                                                            activation=hparams['classifier'],
                                                            num_layers=hparams['layers'],
                                                            num_labels=hparams['num_classes'],
                                                            dropout_pr=hparams['dropout'])
        self.blacklist = ['targets', 'texts']
        

    def training_step(self, batch, batch_nb):
        # I need only part of the batch to train
        train_batch = {k: v.to(self.device) for k, v in batch.items() if k not in self.blacklist}
        outputs = self.model(**train_batch) #labels are inside the train batch
        loss    = outputs.loss
        preds   = torch.argmax(outputs.logits, dim=-1)
        F1_seq_tag = self.compute_f1_seqtag(preds, batch['labels'], display=True)
        F1_score = self.compute_f1_score(batched_preds = preds, 
                                        batched_texts  = batch['texts'], 
                                        batched_labels = batch['labels'], 
                                        batched_targets= batch['targets'], 
                                        display=False)
        self.log('train_macro_F1_seq_tag', F1_seq_tag, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_F1_score',         F1_score,   prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_loss', loss,                   prog_bar=True,  on_epoch=True, on_step=False)
        outputs.loss.backward(retain_graph=True)
        return loss


    def validation_step(self, batch, batch_nb):
        valid_batch = {k: v.to(self.device) for k, v in batch.items() if k not in self.blacklist}
        outputs = self.model(**valid_batch)
        preds   = torch.argmax(outputs.logits, dim=-1)
        # score computed on the Sequence Tagging problem
        F1_seq_tag = self.compute_f1_seqtag(preds, batch['labels'], display=True)
        self.log('macro_F1_seq_tag', F1_seq_tag, prog_bar=False)
        # score computed on the actual task
        F1_score = self.compute_f1_score(batched_preds = preds, 
                                        batched_texts  = batch['texts'], 
                                        batched_labels = batch['labels'], 
                                        batched_targets= batch['targets'], 
                                        display=True)
        self.log('F1_score', F1_score, prog_bar=True)
        self.log('val_loss', outputs.loss, prog_bar=False, on_epoch=True, on_step=False)
        return F1_score
    

    def compute_f1_seqtag(self, preds: Tensor, labels: Tensor, display: bool = False) -> FloatTensor:
        """
        Function to compute the µ, macro, per-class F1 score on the Sequence Tagging problem (PoS, NER, ...) \n
        """
        preds    = preds.view( -1)
        labels   = labels.view(-1)
        indices  = labels != self.hparams['l_vocab'][self.hparams['pad_token']]
        macro_f1 = f1_score(labels[indices].tolist(),
                            preds[ indices].tolist(), 
                            average="macro",
                            zero_division=0)
        if display:
            micro_f1 = f1_score(labels[indices].tolist(), preds[indices].tolist(), average="micro", zero_division=0)
            class_f1 = f1_score(labels[indices].tolist(), preds[indices].tolist(), labels=range(len(self.hparams['l_vocab'])), average=None, zero_division=0)
            print_seqtag_results(macro_f1, micro_f1, class_f1, self.hparams['l_vocab_itos'])
        return FloatTensor([macro_f1], device=self.device)  # required in order to log it with cuda on 


    def process_preds_task_a(self, batched_preds: Tensor, batched_texts: Tensor, batched_labels: Tensor , display:bool = False) -> List[Dict]:
        """
        Take the output of the model and process it to be the output for task A\n
        Starting from the tagged sequence, it builds the set of terms that is the actual prediction
        """
        result = []
        þ = self.hparams['l_vocab_itos']
        if display and batched_labels is not None:
            for predictions, text_vector, labels in zip(batched_preds, batched_texts, batched_labels):
                print('-'*45)  
                print("|GOLD\t\t|PRED\t\t|WORD")
                print('-'*45)  
                for i in range(len(text_vector)):
                    if text_vector[i] == self.hparams['pad_token']:
                        break
                    print(f"{þ[labels[i].item()]}    \t|{þ[predictions[i].item()]}  \t|{text_vector[i]}")

        for predictions, text_vector in zip(batched_preds, batched_texts):
            preds = []
            aux = ''

            for i in range(len(text_vector)):
                if  text_vector[i] == self.hparams['sep_token'] or text_vector[i] == self.hparams['cls_token']:
                    continue

                elif text_vector[i] == self.hparams['pad_token']:
                    break
                
                else:
                    ø = predictions[i].item()
                    #print("PREDICTED LABEL", ø)
                    if 'roberta' in self.hparams['bert_type']:
                        if þ[ø] == BEGINNING:
                            if aux != '':                       # if the buffer is not empty
                                preds.append([aux.strip()])     # append it
                            word = text_vector[i]               # retrieve next word
                            aux = word.replace('Ġ', ' ')        # replace any starting `Ġ` with a space and set it as the first word
                        elif þ[ø] == INSIDE:
                            word = text_vector[i]               # retrieve next word
                            word = word.replace('Ġ', ' ')       # replace any starting `Ġ` with a space (if any)
                            aux += word                         # add word to the string
                        else: 
                            if aux != '':                       # if there's something in the buffer
                                preds.append([aux.strip()])     # append it 
                                aux = ''                        # and clear the variable
                    else:
                        if þ[ø] == BEGINNING:
                            if aux != '':           
                                preds.append([aux.strip()])
                            word = text_vector[i]
                            aux = word.replace('##', '') + ' '
                        
                        elif þ[ø] == INSIDE:
                            word = text_vector[i]
                            if word.startswith('##')     or word.startswith("'")  or \
                               aux.strip().endswith("'") or aux.strip().endswith("$"):
                                aux = aux.strip() 
                            aux += word.replace('##', '') + ' '

                        
                        else:   # outside and padding
                            if aux != '':               # if there's something in the buffer, I append it and clear the variable
                                preds.append([aux.strip()])
                                aux = ''

            # remove empty strings, if any survives
            while '' in preds:
                preds.remove('')
            #print("text:", text_vector)
            #print(">> preds:", preds)
            result.append({'targets' : preds})
        return result


    def compute_f1_score(self, batched_preds: Tensor, batched_texts: Tensor, batched_labels: Tensor, batched_targets: Tensor, display:bool = False) -> FloatTensor:
        """
        Computes the F1 score on the actual task A      \n
        First it process the predictions in the batch   \n
        Then it evaluates them
        """
        results = self.process_preds_task_a(batched_preds, batched_texts, batched_labels, display=display)
        #if self.hparams['uncased']:
        #    self.sanitize_results(data=batched_texts, results=results)
        # >>> useless, as I train with uncased data when uncased is True;
        # in fact, the sentences are encoded all with the uncased version of BERT
        # I only need to sanitize the results when I'm doing the predicting test
        # cause I will test them against a dataset which will not be lowercase.
        return FloatTensor([eval_task_A(batched_targets, results, display=display)], device=self.device)


    @torch.no_grad()
    def predict_task_a(self, data: List[Dict]):
        print(">> Predicting task A ...")
        results = []
        progress_bar = tqdm(range(len(data)))
        tokenizer = self.hparams.tokenizer
        for elem in data:
            elem['text'] = elem['text'].replace('\xa0', ' ')       # deals with some strange cases
            elem['text'] = elem['text'].replace('\u00a0', ' ')     # deals with some strange cases
            
            inputs = tokenizer(elem['text'],
                                return_tensors='pt',
                                padding=False,
                                truncation=True)
            
            if 'roberta' in self.hparams['bert_type']:
                text = tokenizer.tokenize(elem['text'])
                text.insert(        0, self.hparams['cls_token'])
                text.insert(len(text), self.hparams['sep_token'])
                assert len(text) == len(tokenizer.encode(elem['text'], add_special_tokens=True))
            else:
                text  = [tokenizer.tokenize(elem['text'], add_special_tokens=True)]

            res = self.single_predict_task_a(inputs, [text])
            results.append(res[0])
            progress_bar.update(1)

        if self.hparams['uncased']:
            self.sanitize_results(data, results)
        return results


    def sanitize_results(self, data: List[Dict], results: List[Dict]):
        """
        Sanitizes the string given in output from the net\n
        when an uncased BERT is used for tokenization
        """ 
        def tokenize_line(line, pattern='\s'):
            '''tokenizes a line, returns a list of words'''
            line = re.sub('[\.,:;!@#$\(\)\&\\<>]', ' ', line)
            return [word for word in re.split(pattern, line) if word]

        for elem, res in zip(data, results):
            text_sample = elem['text']
            listed_text = tokenize_line(text_sample)

            for pred in res['targets']:
                term = pred[0]

                if term not in text_sample:
                    # to deal with some particular cases
                    term = term.replace(' - ', '-')     # 'touch - pad'             >>      'touch-pad'
                    term = term.replace(' / ', '/')     # 'driver / application'    >>      'driver/application'
                    term = term.replace(' " ', '" ')    # '22 " Monitor'            >>      '22" Monitor'
                    listed_term = tokenize_line(term)
                    
                    for i in range(len(listed_text)):
                        word = listed_text[i]
                        curr = listed_term[0]
                
                        if curr == word.lower():
                            if len(listed_text) >= i + len(listed_term):
                                aux1 = ''
                                aux2 = ''
                                for j in range(len(listed_term)):
                                    aux1 += listed_text[i+j] + ' '
                                    aux2 += listed_term[j]   + ' '   

                                if aux1.strip().lower() == aux2.strip():
                                    pred[0] = aux1.strip()
                                
                                aux2 = aux2.strip(' ') + '/'                # another special case: '8GB RAM/'
                                if aux1.strip().lower() == aux2.strip():
                                    pred[0] = aux1.strip().strip('/')
        return results 


    @torch.no_grad()
    def single_predict_task_a(self, inputs, texts) -> List[Dict]:
        self.model.eval()
        inputs = {k: v.to(self.hparams.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        preds   = torch.argmax(outputs.logits, dim=-1)
        return self.process_preds_task_a(batched_preds=preds,
                                        batched_texts=texts,
                                        batched_labels=None,
                                        display=False)


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams['learning_rate'])