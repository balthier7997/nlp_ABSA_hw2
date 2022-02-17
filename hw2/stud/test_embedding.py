import torch
import numpy as np
import pytorch_lightning as pl

from torch import nn
from tqdm import tqdm
from torch.cuda import FloatTensor
from dataset_class import TermIdentificationDataset



def load_emb(embedding_path):
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
            mapping[splitted[0]] = FloatTensor([float(i) for i in splitted[1:]])
    print(f">> Loaded {len(mapping)} embeddings.")
    return mapping

glove_path      = "hw2/stud/glove/glove.6B.200d.txt"
laptop_path     = "hw2/stud/domain_embedding/laptop_emb.vec"
restaurant_path = "hw2/stud/domain_embedding/restaurant_emb.vec"

glov_mapping = load_emb(glove_path)
lapt_mapping = load_emb(laptop_path)
rest_mapping = load_emb(restaurant_path)

lapt_keys = set(lapt_mapping.keys())
rest_keys = set(rest_mapping.keys())
glov_keys = set(glov_mapping.keys())

difference1 = lapt_keys.difference(rest_keys)
difference2 = lapt_keys.difference(glov_keys)

print(f"Glove words         : {len(glov_mapping)}")
print(f"Laptop words        : {len(lapt_mapping)}")
print(f"Restaurant words    : {len(rest_mapping)}")
print(f"Words in laptop, not in restaurants: {len(difference1)}")
print(f"Words in laptop, not in glove      : {len(difference2)}")
print(f"Words in laptop, not in restaurants - Words in laptop, not in glove: {len(difference1.difference(difference2))}")