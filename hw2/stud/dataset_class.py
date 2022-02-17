import re
import json
import torch
import numpy as np

from pprint           import pprint
from tqdm             import tqdm
from typing           import Dict, List
from pprint           import pprint
from collections      import Counter
from torchtext.vocab  import Vocab
from torch.utils.data import Dataset
try:
    from utils  import *
except:
    from stud.utils  import *

######################### task A ######################### 

class TermIdentificationDataset(Dataset):

    def __init__(self, input_files, size=-1, device='cuda', debug=False, vocab=None):
        self.input_files = input_files
        self.vocab      = vocab
        self.device     = device
        self.debug      = debug
        self.unk_token  = UNK_TOKEN_ROBERTA
        self.pad_token  = PAD_TOKEN_ROBERTA
        self.size       = size
        self.data       = []
        self.l_vocab    = { PAD_TOKEN_ROBERTA   : 0,
                            BEGINNING           : 1,
                            INSIDE              : 2,
                            OUTSIDE             : 3}

        self.l_vocab_itos = {v : k for k, v in self.l_vocab.items()}
        print(self.l_vocab.stoi) if self.debug else 0

        raw_data = []
        for file in self.input_files:
            with open(file, 'r') as f:
                raw_data.extend(json.load(f))
        
        if self.debug:
            print("Data elem:")
            self.print_json_object(raw_data[0]) if self.debug else 0

        # I add the term vector to each elem of the dataset
        # it will be the 'label' for task A
        # While doing so I also build the vocabulary using a Counter()
        max_len = 0
        if self.vocab is None:
            counter = Counter()
        i = 0
        for elem in raw_data:
            if self.size > 0 and i == self.size:
                break
            words_vector, terms_vector = self.build_terms_vector(elem)
            elem['words_vector'] = words_vector
            elem['terms_vector'] = terms_vector

            # the max len is the lenght to pad all sentences
            if len(elem['words_vector']) > max_len:
                max_len = len(elem['words_vector'])
            # counter update
            if self.vocab is None:  
                for word in elem['words_vector']:
                    counter[word]+=1
            
            i += 1
        if self.vocab is None:
            print(">> Creating new vocab...")
            self.vocab = Vocab(counter, specials=[self.pad_token, self.unk_token], min_freq=1)
        if self.debug:    
            print("Data elem (after adding term vector):")
            self.print_json_object(raw_data[5])
    
        # encoding the sentences
        i = 0
        for elem in raw_data:
            if self.size > 0 and i == self.size:
                break
            l = len(elem['words_vector'])

            for _ in range(l, max_len):
                elem['words_vector'].append(self.pad_token)
                elem['terms_vector'] = np.append(elem['terms_vector'], self.l_vocab[self.pad_token])
            
            elem['terms_vector'] = torch.LongTensor(elem['terms_vector']).to(self.device)
            elem['idxs_vector']  = TermIdentificationDataset.words2indexes(elem['words_vector'], self.pad_token, self.vocab, self.device)
            assert  len(elem['words_vector']) == len(elem['terms_vector']) == len(elem['idxs_vector']) == max_len
            
            i += 1
        if self.debug:
            print("Data elem (after padding and encoding):")
            self.print_json_object(raw_data[0])     if self.debug else 0
            print("Pad token vocab @", self.vocab[  self.pad_token])
            print("Pad token label @", self.l_vocab[self.pad_token])
        print(">> Sentence pad length:\t", max_len)

        # return only part of the dataset if so requested
        if self.size < 0:
            self.data = raw_data
        else:
            i = 0
            for elem in raw_data:
                if i == self.size:
                    break
                self.data.append(elem)
                i += 1
        

    @staticmethod
    def tokenize_line(line, pattern='\W'):
        '''tokenizes a line, returns a list of words'''
        # I insert a space otherwise words such as 'xx-yyyy' becames 'xxyyyy'
        # which, in most cases, isn't proper english and appears only once or twice
        line = re.sub('[\.,:;!@#$\(\)\-&\\<>/]', ' ', line)
        return [word for word in re.split(pattern, line) if word]


    def build_terms_vector(self, elem):
        listed_sentence = TermIdentificationDataset.tokenize_line(elem['text'])
        term_vector = np.full_like(listed_sentence, fill_value=self.l_vocab[OUTSIDE], dtype=int)
        
        if self.debug:
            print(elem['text'])
            print(listed_sentence)

        for t in elem['targets']:
            listed_targets = TermIdentificationDataset.tokenize_line(t[1])
            print(listed_targets, t[2]) if self.debug else 0

            for i in range(len(listed_sentence)):
                if listed_sentence[i] == listed_targets[0]:
                    term_len = len(listed_targets)
                    aux1 = ""
                    aux2 = ""
                    for j in range(term_len):
                        print("i + j;", i, '+', j) if self.debug else 0
                        aux1 += listed_sentence[i+j] + ' '
                        aux2 += listed_targets[j] + ' '
                    if self.debug:
                        print(aux1)
                        print(aux2)
                    if aux1 == aux2:
                        for j in range(term_len):
                            term_vector[i+j]      = self.l_vocab[INSIDE]
                        term_vector[i]            = self.l_vocab[BEGINNING]
                        break               # used to optimize things and to avoid cases in which 
                                            # a prefix of the terms appears near the end causing IndexOutOfRange
        print(term_vector) if self.debug else 0
        return listed_sentence, term_vector


    @staticmethod
    def words2indexes(words_vector, pad_token, vocab, device):
        þ = np.zeros_like(words_vector, dtype=int)
        for i in range(len(words_vector)):
            if words_vector[i] == pad_token:   # it's already padded, it is initialized at 0 -> the encodind of the pad token
                break
            þ[i] = vocab[words_vector[i]]
        return torch.LongTensor(þ).to(device)


    @staticmethod
    def collate_fn(batch):
        inputs  = torch.stack([elem['idxs_vector' ] for elem in batch], 0)
        labels  = torch.stack([elem['terms_vector'] for elem in batch], 0)
        texts   = []
        targets = []
        for elem in batch:
            texts.append(elem['words_vector'])
            targets.append({'targets' : elem['targets']})
        assert inputs.size() == labels.size()
        return inputs, labels, texts, targets


    def print_json_object(self, obj):
        print("\n")
        for k,v in obj.items():
            print(f">> {k}:")
            print(v)
        print()


    def __getitem__(self, idx):
        return self.data[idx]


    def __len__(self):
        return len(self.data)


########### for BERT model
class TermIdentificationDatasetBERT(Dataset):

    def __init__(self, input_files, tokenizer, rm_stopwords, stopset, size=-1, roberta=False, device='cuda', debug=False):
        self.input_files = input_files

        self.size       = size
        self.debug      = debug
        self.device     = device
        self.roberta    = roberta
        self.stopset    = stopset
        self.tokenizer  = tokenizer 
        self.vocab      = tokenizer.get_vocab()
        self.vocab_rev  = {v : k for k, v in self.vocab.items()}
        self.unk_token  = UNK_TOKEN_ROBERTA   if roberta  else  PAD_TOKEN_BERT
        self.pad_token  = PAD_TOKEN_ROBERTA   if roberta  else  UNK_TOKEN_BERT
        self.cls_token  = CLS_TOKEN_ROBERTA   if roberta  else  CLS_TOKEN_BERT
        self.sep_token  = SEP_TOKEN_ROBERTA   if roberta  else  SEP_TOKEN_BERT
        self.remove_stopwords = rm_stopwords
        
        self.l_vocab    = { self.pad_token  : 0,
                            BEGINNING       : 1,
                            INSIDE          : 2,
                            OUTSIDE         : 3 }
        self.l_vocab_itos = {v : k for k, v in self.l_vocab.items()}

        self.data = []
        for file in self.input_files:
            with open(file, 'r') as f:
                self.data.extend(json.load(f))
        
        for elem in self.data:
            elem['text'] = elem['text'].replace('\xa0', ' ')       # deals with some strange cases
            elem['text'] = elem['text'].replace('\u00a0', ' ')     # deals with some strange cases
            
        if self.remove_stopwords:
            for elem in self.data:
                elem['text'] = elem['text'].replace('\xa0', ' ')     # deals with some strange cases
                text = [word for word in elem['text'].split(' ') if word.lower() not in self.stopset]
                elem['text'] = ''
                for token in text:
                    elem['text'] += token + ' '
                elem['text'] = elem['text'].strip(' ')


    def collate_fn_BERT(self, batch): 
        # I add the terms vector to each elem of the dataset (label of task A)
        inputs = self.tokenizer([elem['text'] for elem in batch], 
                                truncation=True,
                                padding='longest',
                                return_tensors='pt')

        # build the labels
        terms_vectors = np.full_like(inputs['input_ids'], dtype=int,
                                    fill_value=self.l_vocab[self.pad_token])

        batch_size, batch_seq_len = inputs['input_ids'].size()

        if self.roberta:
            for elem_idx in range(batch_size):
                elem    = batch[elem_idx]
                current = inputs['input_ids'][elem_idx]
                
                roberta_targets = []                    # need to add also the target with a space before 
                for tar in elem['targets']:             # because roberta is encodes differently 'pasta' and ' pasta'
                    roberta_targets.append(tar[1])
                    roberta_targets.append(' ' + tar[1])

                for tar in roberta_targets:
                    encoded_targets = self.tokenizer.encode(tar, add_special_tokens=False)     # don't need [CLS] and [SEP] here
                    
                    # I need to check that the whole sequence of terms is the same
                    # there may be cases in which a prefix appears more than once in a sentence
                    for i in range(batch_seq_len):
                        if current[i] == encoded_targets[0]:   # i-th word equal to first one of the term
                            term_len = len(encoded_targets)
                            if len(current) - i >= term_len:    # prevents IndexOutOfError for prefix matches
                                aux1 = []
                                aux2 = []
                                for j in range(term_len):
                                    aux1.append(current[i+j].item())
                                    aux2.append(encoded_targets[j])
                                if aux1 == aux2:                            # I check all the sequence long 'term_len'
                                    for j in range(term_len):
                                        terms_vectors[elem_idx][i+j] = self.l_vocab[INSIDE]
                                    terms_vectors[elem_idx][i]       = self.l_vocab[BEGINNING]
                                    continue
                
                # tag the outside tokens
                for i in range(batch_seq_len):
                    if current[i] == 1:         # roberta's  encoding for pad token:
                        break
                    elif current[i] in (0, 2):  # roberta's encoding for  cls and sep token
                        continue
                    else:
                        if terms_vectors[elem_idx][i] == self.l_vocab[self.pad_token]:
                            terms_vectors[elem_idx][i] = self.l_vocab[OUTSIDE]
        
        else:
            for elem_idx in range(batch_size):
                elem = batch[elem_idx]
                #print(elem['text'])
                for t in elem['targets']:
                    #print(">>", t[1])
                    encoded_targets = self.tokenizer.encode(t[1], add_special_tokens=False)     # don't need [CLS] and [SEP] here
                    
                    # I need to check that the whole sequence of terms is the same
                    # there may be cases in which a prefix appears more than once in a sentence
                    for i in range(batch_seq_len):
                        current = inputs['input_ids'][elem_idx]
                        if current[i] == encoded_targets[0]:   # i-th word equal to first one of the term
                            term_len = len(encoded_targets)
                            if len(current) - i >= term_len:    # prevents IndexOutOfError for prefix matches
                                aux1 = []
                                aux2 = []
                                for j in range(term_len):
                                    aux1.append(current[i+j].item())
                                    aux2.append(encoded_targets[j])
                                if aux1 == aux2:                            # I check all the sequence long 'term_len'
                                    for j in range(term_len):
                                        terms_vectors[elem_idx][i+j] = self.l_vocab[INSIDE]
                                    terms_vectors[elem_idx][i]       = self.l_vocab[BEGINNING]
                                    continue
                
                # tag the outside tokens
                for i in range(batch_seq_len):
                    if inputs['input_ids'][elem_idx][i] == self.vocab[self.pad_token]:
                        break
                    elif inputs['input_ids'][elem_idx][i] == self.vocab[self.cls_token] or \
                            inputs['input_ids'][elem_idx][i] == self.vocab[self.sep_token]:
                        continue
                    else:
                        if terms_vectors[elem_idx][i] != self.l_vocab[BEGINNING] and \
                        terms_vectors[elem_idx][i] != self.l_vocab[INSIDE]:
                            terms_vectors[elem_idx][i] = self.l_vocab[OUTSIDE]
            
        # build the batch for bert
        input_ids = torch.stack([elem.squeeze()  for elem in inputs['input_ids'     ]], 0)
        token_ids = torch.stack([elem.squeeze()  for elem in inputs['token_type_ids']], 0)  if not self.roberta else 0
        atte_mask = torch.stack([elem.squeeze()  for elem in inputs['attention_mask']], 0)

        terms_vectors = torch.LongTensor(terms_vectors).to(self.device)
        label_ids = torch.stack([elem.squeeze()  for elem in terms_vectors], 0)
        
        words_vectors = []
        if self.roberta:
            for elem in batch:
                aux = self.tokenizer.tokenize(elem['text'])
                aux.insert(0,        self.cls_token)
                aux.insert(len(aux), self.sep_token)
                        
                assert len(aux) == len(self.tokenizer.encode(elem['text'], add_special_tokens=True))
                words_vectors.append(aux)
        else:
            words_vectors.extend(self.tokenizer.tokenize(elem['text'], add_special_tokens=True) for elem in batch)

        targets = []
        targets.extend({'targets' : elem['targets']} for elem in batch)
        
        # modify attention_mask
        # >> don't do it, poorer results <<
        # for mask in atte_mask:
        #     for i in range(len(mask)):
        #         # set CLS token to be ignored
        #         if i == 0:              
        #             mask[i] = 0
        #         # set PAD token to be ignored and terminate the inner loop
        #         if i+1 < len(mask) and mask[i] == 1 and mask[i+1] == 0:
        #             mask[i] = 0
        #             break

        if self.debug:
            print("input_ids", input_ids[0])
            #print("token_ids", token_ids[0])   # useless, I'm using only one sentence 
            print("atte_mask", atte_mask[0])
            print("label_ids", label_ids[0])
            print()
            #print("words_vec", words_vectors)
            #print("targets  ", targets)
        
        if self.roberta:
            assert input_ids.size() == atte_mask.size() == label_ids.size()
        else:
            assert input_ids.size() == token_ids.size() == atte_mask.size() == label_ids.size()
        
        res = { 'input_ids'      : input_ids, 
                'attention_mask' : atte_mask,
                'labels'         : label_ids,
                'texts'          : words_vectors, 
                'targets'        : targets}

        if not self.roberta:
            res['token_type_ids'] = token_ids

        return res


    def __getitem__(self, idx):
        return self.data[idx]


    def __len__(self):
        return len(self.data)


######################### task B ######################### 


class TermPolarityDatasetBERT(Dataset):

    def __init__(self, input_files, tokenizer, stopwords, stopset, absent_token='', first='term', roberta=False, size=-1, device='cuda', debug=False):
        self.input_files = input_files

        self.device     = device
        self.debug      = debug
        self.tokenizer  = tokenizer 
        self.vocab      = tokenizer.get_vocab()
        self.vocab_rev  = {v : k for k, v in self.vocab.items()}
        self.size       = size
        self.first      = first
        self.remove_stopwords = stopwords
        self.stopset = stopset
        self.absent_token = absent_token
        self.roberta = roberta

        self.l_vocab = {NEUTRAL  : 0,
                        POSITIVE : 1,
                        NEGATIVE : 2,
                        CONFLICT : 3,
                        ABSENT   : 4}

        self.l_vocab_itos  = {v : k for k, v in self.l_vocab.items()}

        self.raw_data = []
        for file in self.input_files:
            with open(file, 'r') as f:
                self.raw_data.extend(json.load(f))
        
        self.data = TermPolarityDatasetBERT.preprocessing_task_b(raw_data=self.raw_data, 
                                                                l_vocab=self.l_vocab,
                                                                testing=False,
                                                                device=self.device,
                                                                absent_token=self.absent_token)
                
        if self.remove_stopwords:
            for elem in self.data:
                elem['text'] = elem['text'].replace('\xa0', ' ')     # deals with some strange cases
                text = [word for word in elem['text'].split(' ') if word.lower() not in self.stopset]
                elem['text'] = ''
                for token in text:
                    elem['text'] += token + ' '
                elem['text'] = elem['text'].strip(' ')


    @staticmethod
    def preprocessing_task_b(raw_data: List[Dict], l_vocab, absent_token='', testing=False, device='cpu'):
        """
        Prepare the dataset by adding the aspect term in the sentence\n
        If a sentence has more than one, I duplicate that sentence, each time with a differente aspect term\n
        If a sentence has none, then I'll append an empty string (the absent dummy class takes care of this case)
        """
        data = []
        print("Preprocessing the input for task B ...")
        for elem in raw_data:
            text    = elem['text']
            targets = elem['targets']
            
            if len(targets) == 0:
                targets.append([[0,-1], absent_token, ABSENT])

            uusi = []
            for target in targets:
                term  = target[1]
                if not testing:
                    label = target[2]
                
                þ = {}
                þ['text' ] = text
                þ['term' ] = term
                þ['targets'] = targets
                if not testing:
                    þ['label'] = torch.LongTensor([l_vocab[label]]).to(device)
                uusi.append(þ)
            data.extend(uusi) 
        return data


    @staticmethod
    def test_dataset_builder(input_files) -> List[Dict]:
        """
        Reads the dataset from the input files and return a List[Dict] containing the same format
        used in the evaluation with docker\n
        """
        raw_data = []
        for file in input_files:
            with open(file, 'r') as f:
                raw_data.extend(json.load(f))
        data = []

        for elem in raw_data:
            data.append({'text'   : elem['text'],
                        'targets' : elem['targets']})
        return data


    def collate_fn(self, batch):
        if self.first == 'term':
            inputs = self.tokenizer([elem['term'] for elem in batch],
                                    [elem['text'] for elem in batch], 
                                    truncation=True,
                                    padding='longest',
                                    return_tensors='pt')
        elif self.first == 'text':
            inputs = self.tokenizer([elem['text'] for elem in batch],
                                    [elem['term'] for elem in batch], 
                                    truncation=True,
                                    padding='longest',
                                    return_tensors='pt')
        else:
            raise NotImplementedError

        input_ids = torch.stack([elem.squeeze()  for elem in inputs['input_ids'     ]], 0)
        token_ids = torch.stack([elem.squeeze()  for elem in inputs['token_type_ids']], 0)  if not self.roberta else 0
        atte_mask = torch.stack([elem.squeeze()  for elem in inputs['attention_mask']], 0)
        labels    = torch.stack([elem['label']   for elem in batch], 0)
        terms   = []
        texts   = []
        targets = []
        
        curr = ''
        for elem in batch:
            terms.append(elem['term'])
            texts.append(elem['text'])
            if elem['text'] != curr:
                targets.append({'targets' : elem['targets']})
            curr = elem['text']
        
        res = { 'input_ids'      : input_ids, 
                'attention_mask' : atte_mask,
                'labels'         : labels,
                'texts'          : texts,
                'terms'          : terms,
                'targets_list'   : targets}
        if not self.roberta:
            res['token_type_ids'] = token_ids
        
        return res


    @staticmethod
    def test_collate_fn(batch, tokenizer, first='term', roberta=False):
        if first == 'term':
            inputs = tokenizer( [elem['term'] for elem in batch],
                                [elem['text'] for elem in batch], 
                                truncation=True,
                                padding='longest',
                                return_tensors='pt')
        elif first == 'text':
            inputs = tokenizer( [elem['text'] for elem in batch],
                                [elem['term'] for elem in batch], 
                                truncation=True,
                                padding='longest',
                                return_tensors='pt')
        else:
            raise NotImplementedError
        input_ids = torch.stack([elem.squeeze()  for elem in inputs['input_ids'     ]], 0)
        token_ids = torch.stack([elem.squeeze()  for elem in inputs['token_type_ids']], 0)  if not roberta else 0
        atte_mask = torch.stack([elem.squeeze()  for elem in inputs['attention_mask']], 0)
        terms   = []
        texts   = []
        targets = []
        
        curr = ''
        for elem in batch:
            terms.append(elem['term'])
            texts.append(elem['text'])
            if elem['text'] != curr:
                targets.append({'targets' : elem['targets']})
            curr = elem['text']

        res = { 'input_ids'      : input_ids, 
                'attention_mask' : atte_mask,
                'texts'          : texts,
                'terms'          : terms,
                'targets_list'   : targets}
        if not roberta:
            res['token_type_ids'] = token_ids

        return res


    def __getitem__(self, idx):
        return self.data[idx]


    def __len__(self):
        return len(self.data)


class TermPolarityDatasetBERT_noDummy(Dataset):

    def __init__(self, input_files, tokenizer, size=-1, device='cuda', debug=False):
        self.input_files = input_files

        self.device     = device
        self.debug      = debug
        self.tokenizer  = tokenizer 
        self.vocab      = tokenizer.get_vocab()
        self.vocab_rev  = {v : k for k, v in self.vocab.items()}
        self.size       = size
        
        self.l_vocab = {NEUTRAL  : 0,
                        POSITIVE : 1,
                        NEGATIVE : 2,
                        CONFLICT : 3}

        self.l_vocab_itos  = {v : k for k, v in self.l_vocab.items()}

        self.raw_data = []
        for file in self.input_files:
            with open(file, 'r') as f:
                self.raw_data.extend(json.load(f))
        
        self.data = TermPolarityDatasetBERT_noDummy.preprocessing_task_b(raw_data=self.raw_data, 
                                                                l_vocab=self.l_vocab,
                                                                testing=False,
                                                                device=self.device)

    def get_raw_data(self):
        return self.raw_data


    @staticmethod
    def preprocessing_task_b(raw_data: List[Dict], l_vocab, testing=False, device='cpu'):
        """
        Prepare the dataset by adding the aspect term in the sentence\n
        If a sentence has more than one, I duplicate that sentence, each time with a differente aspect term\n
        If a sentence has none, I just skip it and I'll take care in the processing to add it
        """
        data = []
        print("Preprocessing the input for task B ...")
        count = 0
        for elem in raw_data:
            if len(elem['targets']) > 0:
                uusi    = []
                text    = elem['text']
                targets = elem['targets']
                
                for target in targets:
                    term  = target[1]
                    if not testing:
                        label = target[2]
                    
                    þ = {}
                    þ['text' ] = text
                    þ['term' ] = term 
                    þ['targets'] = targets
                    if not testing:
                        þ['label'] = torch.LongTensor([l_vocab[label]]).to(device)
                    uusi.append(þ)

                data.extend(uusi) 
            else:
                count += 1
        print(f">>> Skipped {count} term-void sentences.")
        print(f">> data len is {len(data)}")
        return data


    @staticmethod
    def test_preprocessing_task_b(raw_data: List[Dict], l_vocab, testing=False, device='cpu'):
        """
        Prepare the dataset by adding the aspect term in the sentence\n
        If a sentence has more than one, I duplicate that sentence, each time with a differente aspect term\n
        If a sentence has none, I add an empty string
        """
        data = []
        print("Preprocessing the input for task B ...")
        count = 0
        for elem in raw_data:
            uusi    = []
            text    = elem['text']
            targets = elem['targets']
            
            if len(targets) == 0:
                count += 1
                targets.append([[0,-1], '', ABSENT])
            
            for target in targets:
                term  = target[1]
                if not testing:
                    label = target[2]
                
                þ = {}
                þ['text' ] = text
                þ['term' ] = term 
                þ['targets'] = targets
                if not testing:
                    þ['label'] = torch.LongTensor([l_vocab[label]]).to(device)
                uusi.append(þ)

            data.extend(uusi) 
        print(f">>> Added {count} absent flags")
        return data


    @staticmethod
    def test_dataset_builder(input_files) -> List[Dict]:
        """
        Reads the dataset from the input files and return a List[Dict] containing the same format
        used in the evaluation with docker\n
        `preds_a` is used when doing task a+b
        """
        raw_data = []
        for file in input_files:
            with open(file, 'r') as f:
                raw_data.extend(json.load(f))
        data = []

        for elem in raw_data:
            data.append({'text'    : elem['text'],
                         'targets' : elem['targets']})
        return data


    def collate_fn(self, batch):
        inputs = self.tokenizer([elem['term'] for elem in batch],
                                [elem['text'] for elem in batch], 
                                truncation=True,
                                padding='longest',
                                return_tensors='pt')
        input_ids = torch.stack([elem.squeeze()  for elem in inputs['input_ids'     ]], 0)
        token_ids = torch.stack([elem.squeeze()  for elem in inputs['token_type_ids']], 0)
        atte_mask = torch.stack([elem.squeeze()  for elem in inputs['attention_mask']], 0)
        labels    = torch.stack([elem['label']   for elem in batch], 0)
        terms   = []
        texts   = []
        targets = []
        
        curr = ''
        for elem in batch:
            terms.append(elem['term'])
            texts.append(elem['text'])
            if elem['text'] != curr:
                targets.append({'targets' : elem['targets']})
            curr = elem['text']

        return {'input_ids'      : input_ids, 
                'token_type_ids' : token_ids,
                'attention_mask' : atte_mask,
                'labels'         : labels,
                'texts'          : texts,
                'terms'          : terms,
                'targets_list'   : targets}


    @staticmethod
    def test_collate_fn(batch, tokenizer):
        inputs = tokenizer( [elem['term'] for elem in batch],
                            [elem['text'] for elem in batch], 
                            truncation=True,
                            padding='longest',
                            return_tensors='pt')
        input_ids = torch.stack([elem.squeeze()  for elem in inputs['input_ids'     ]], 0)
        token_ids = torch.stack([elem.squeeze()  for elem in inputs['token_type_ids']], 0)
        atte_mask = torch.stack([elem.squeeze()  for elem in inputs['attention_mask']], 0)
        terms   = []
        texts   = []
        targets = []
        
        curr = ''
        for elem in batch:
            terms.append(elem['term'])
            texts.append(elem['text'])
            if elem['text'] != curr:
                targets.append({'targets' : elem['targets']})
            curr = elem['text']

        return {'input_ids'      : input_ids, 
                'token_type_ids' : token_ids,
                'attention_mask' : atte_mask,
                'texts'          : texts,
                'terms'          : terms,
                'targets_list'   : targets}


    def __getitem__(self, idx):
        return self.data[idx]


    def __len__(self):
        return len(self.data)

