import json
import torch

from pprint           import pprint
from tqdm             import tqdm
from typing           import Dict, List
from pprint           import pprint
from torch.utils.data import Dataset

try:
    from utils  import *
except:
    from stud.utils  import *

######################### task C #########################


class CategoryExtractionDatasetBERT(Dataset):
    def __init__(self, input_file, tokenizer, roberta=False, size=-1, device='cuda', debug=False):
        self.size       = size
        self.debug      = debug
        self.device     = device
        self.roberta    = roberta
        self.tokenizer  = tokenizer 
        self.input_file = input_file
        self.vocab      = tokenizer.get_vocab()
        self.vocab_rev  = {v : k for k, v in self.vocab.items()}
        self.l_vocab    = { FOOD     : 0,
                            MISC     : 1,
                            SERVICE  : 2,
                            AMBIENCE : 3,
                            PRICE    : 4}
        self.l_vocab_itos  = {v : k for k, v in self.l_vocab.items()}

        self.raw_data = []
        for file in self.input_file:
            with open(file, 'r') as f:
                self.raw_data.extend(json.load(f))
                
        self.data = CategoryExtractionDatasetBERT.preprocessing_task_c(raw_data=self.raw_data, 
                                                                        l_vocab=self.l_vocab,
                                                                        testing=False,
                                                                        device=self.device)


    @staticmethod
    def preprocessing_task_c(raw_data: List[Dict], l_vocab, testing=False, device='cpu'):
        """
        Prepare the dataset by adding the label at each sentence\n
        The label will be a vector of zeroes, with ones in the position of the category of that sentence\n
        More than one '1' (and so more than one category) are possibles\n
        Returns a List[Dict] having as keys:
        - text : the original text (str)
        - categories : list of categories (`[{'categories': [['anecdotes/miscellaneous', 'positive'], ['food', 'negative']]}, ...]`)
        - label : list of [0,1] with 1 if the category being tested is right, otherwise 0
        """
        data = []
        print("Preprocessing the input of task C ...")        
        for elem in raw_data:
            þ = {}
            þ['text'] = elem['text']
            if 'categories' in elem.keys():
                þ['categories'] = elem['categories']
            if not testing:
                label = [0,0,0,0,0]
                for category in þ['categories']:
                    label[l_vocab[category[0]]] = 1
                þ['label'] = torch.LongTensor(label).to(device)
            data.append(þ) 
        return data


    def collate_fn_wrapper(self, batch):
        return CategoryExtractionDatasetBERT.collate_fn(batch=batch, tokenizer=self.tokenizer, roberta=self.roberta, testing=False)


    @staticmethod
    def collate_fn(batch, tokenizer, roberta=False, testing=False):            
        inputs = tokenizer([elem['text']     for elem in batch], 
                                truncation=True,
                                padding='longest',
                                return_tensors='pt' )
        input_ids = torch.stack([elem.squeeze()  for elem in inputs['input_ids'     ]], 0)
        atte_mask = torch.stack([elem.squeeze()  for elem in inputs['attention_mask']], 0)
        token_ids = torch.stack([elem.squeeze()  for elem in inputs['token_type_ids']], 0)  if not roberta else 0
        labels    = torch.stack([elem['label']   for elem in batch], 0).float()             if not testing else 0
        targets = []
        
        if not testing:
            curr = ''
            for elem in batch:
                if elem['text'] != curr:
                    targets.append({'categories' : elem['categories']})
                curr = elem['text']

        res = { 'input_ids'       : input_ids, 
                'attention_mask'  : atte_mask,
                'categories_list' : targets}     # needed as labels for the evaluation 
        if not roberta:
            res['token_type_ids'] = token_ids
        if not testing:
            res['labels'] = labels
            res['categories_list'] = targets
        return res


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
            data.append({'text'       : elem['text'],
                         'categories' : elem['categories']})
        return data


    def __getitem__(self, idx):
        return self.data[idx]


    def __len__(self):
        return len(self.data)



######################### task D #########################



class CategorySentimentDatasetBERT(Dataset):
    def __init__(self, input_file, tokenizer, roberta=False, stopwords=False, stopset=None, size=-1, device='cuda', first='sentence', debug=False):
        self.raw_data   = []
        self.size       = size
        self.first      = first
        self.debug      = debug
        self.device     = device
        self.roberta    = roberta
        self.stopset    = stopset
        self.stopwords  = stopwords
        self.tokenizer  = tokenizer 
        self.input_file = input_file
        self.vocab      = tokenizer.get_vocab()
        self.vocab_rev  = {v : k for k, v in self.vocab.items()}
        self.l_vocab = {NEUTRAL  : 0,
                        POSITIVE : 1,
                        NEGATIVE : 2,
                        CONFLICT : 3}
        self.l_vocab_itos  = {v : k for k, v in self.l_vocab.items()}
        for file in self.input_file:
            with open(file, 'r') as f:
                self.raw_data.extend(json.load(f))
        
        self.data = CategorySentimentDatasetBERT.preprocessing_task_d(raw_data=self.raw_data, 
                                                                    l_vocab=self.l_vocab,
                                                                    testing=False,
                                                                    device=self.device)


    @staticmethod
    def preprocessing_task_d(raw_data: List[Dict], l_vocab, testing=False, device='cpu'):
        """
        Prepare the dataset by adding the category in the sentence\n
        If a sentence has more than one, I duplicate that sentence, each time with a differente aspect term
        """
        data = []
        print("Preprocessing the input. . .")
        progress_bar = tqdm(range(len(raw_data)))
        for elem in raw_data:
            text       = elem['text']
            categories = elem['categories']
            
            uusi = []
            for couple in categories:
                category = couple[0]                
                þ = {}
                þ['text' ]      = text
                þ['category'  ] = category 
                þ['categories'] = categories
                if not testing:
                    label = couple[1]   # sentiment of the category
                    þ['label'] = torch.LongTensor([l_vocab[label]]).to(device)
                uusi.append(þ)

            data.extend(uusi) 
            progress_bar.update(1)
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
            data.append({'text'      : elem['text'],
                        'categories' : elem['categories']})
        return data


    def collate_fn_wrapper(self, batch):
        return CategorySentimentDatasetBERT.collate_fn(batch, tokenizer=self.tokenizer, roberta=self.roberta, first=self.first, testing=False)


    @staticmethod
    def collate_fn(batch, tokenizer, roberta, first, testing):
        if first == 'text':
            inputs = tokenizer([elem['text']       for elem in batch], 
                                    [elem['category']   for elem in batch],
                                    truncation=True,
                                    padding='longest',
                                    return_tensors='pt')        
        elif first == 'category':
            inputs = tokenizer([elem['category']   for elem in batch],
                                    [elem['text']       for elem in batch], 
                                    truncation=True,
                                    padding='longest',
                                    return_tensors='pt')
        input_ids = torch.stack([elem.squeeze()  for elem in inputs['input_ids'     ]], 0)
        atte_mask = torch.stack([elem.squeeze()  for elem in inputs['attention_mask']], 0)
        token_ids = torch.stack([elem.squeeze()  for elem in inputs['token_type_ids']], 0)  if not roberta else None
        labels    = torch.stack([elem['label']   for elem in batch], 0)                     if not testing else None
        
        texts = []
        categories = []
        categories_list = []
        
        curr = ''
        for elem in batch:
            categories.append(elem['category'])
            texts.append(elem['text'])
            if elem['text'] != curr:
                categories_list.append({'categories' : elem['categories']})
            curr = elem['text']

        res = { 'input_ids'       : input_ids, 
                'attention_mask'  : atte_mask,
                'texts'           : texts,
                'categories'      : categories,
                'categories_list' : categories_list}
        if not roberta:
            res['token_type_ids'] = token_ids
        if not testing:
            res['labels'] = labels
        return res


    def __getitem__(self, idx):
        return self.data[idx]


    def __len__(self):
        return len(self.data)
