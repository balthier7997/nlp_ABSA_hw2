import gc
import nltk
import torch
import argparse
import pytorch_lightning as pl

from torch.utils.data  import DataLoader
from pytorch_lightning import callbacks as clbk

from transformers import AutoTokenizer, RobertaTokenizer

from utils         import *
from model_class_B import TermPolarityModelBERT,   TermPolarityModelBERT_noDummy, TermPolarityModelROBERTA
from dataset_class import TermPolarityDatasetBERT, TermPolarityDatasetBERT_noDummy, NEUTRAL, NEGATIVE, POSITIVE, ABSENT, CONFLICT
from pprint import pprint


gc.collect()
pl.seed_everything(42, workers=True)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")

parser = argparse.ArgumentParser()

parser.add_argument('mode', help="train or test")
parser.add_argument('-f',  '--first', 
                    help="term or text", type=str, default='term')
parser.add_argument('-a',  '--activation', 
                    help="base relu gelu selu tanh mish",
                    default='base')
parser.add_argument('-n',  '--num_layers', 
                    help="1 2 3",
                    type=int,
                    default=1)
parser.add_argument('-m',  '--model', 
                    help="bert-base-cased, bert-base-uncased, roberta-base",
                    default='bert-base-cased')
parser.add_argument('-rs', '--rm_stopwords', 
                    help="whether to remove stopwords",
                    action='store_true',
                    default=False)
parser.add_argument('-dl',  '--no_dummy_label', 
                    help="whether to add the dummy label and token",
                    action='store_false',
                    default=True)
parser.add_argument('-p',  '--pooling', 
                    help="concat-mean, concat-max, reduce-mean, reduce-max, cls",
                    default=None)
parser.add_argument('-l',  '--pooling_layer', 
                    help="from -1 to -12 or an interval as -4:-1",
                    default=None, 
                    nargs='+',
                    type=int)
parser.add_argument('-d', '--dropout', 
                    help="dropout probability",
                    type=float,
                    default=0.1)
parser.add_argument('-lr', '--learning_rate', 
                    help="learning rate",
                    type=float,
                    default=0.1)
parser.add_argument('-hd', '--hidden_dim', 
                    help="hidden dimension of the classifier",
                    type=int,
                    default=768)
args = parser.parse_args()

ACT              = args.activation
MODE             = args.mode
FIRST            = args.first
MODEL            = args.model
POOLING          = args.pooling
DROPOUT          = args.dropout
NUM_LAYERS       = args.num_layers
DUMMY_LABEL      = args.no_dummy_label
POOLER_LAYER     = [elem for elem in args.pooling_layer] if args.pooling_layer is not None else args.pooling_layer
HIDDEN_DIM       = args.hidden_dim
LEARNING_RATE    = args.learning_rate
REMOVE_STOPWORDS = args.rm_stopwords


MAX_EPOCHS       = 15
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE   = 32
train_files      = [TRAIN_FILE_R, TRAIN_FILE_L]
devel_files      = [DEVEL_FILE_R, DEVEL_FILE_L]

logname = f"B_{MODEL.split('-')[0]}-SeqCls_{ACT}-{NUM_LAYERS}_{FIRST}-first_lr={LEARNING_RATE}_drop={DROPOUT}"


if POOLING is not None:
    ABSENT_TOKEN = '.'
    logname += f'_{POOLING}{POOLER_LAYER}'
else:
    ABSENT_TOKEN = ''

if REMOVE_STOPWORDS:
    nltk.download('stopwords', download_dir='./nltk_data')
    nltk.data.path.append('./nltk_data')
    stopset = set(nltk.corpus.stopwords.words('english'))
    logname += f'_rm-stops'
else:
    stopset = {}

if not DUMMY_LABEL:
    logname += '_no-dummy'

if MODE == 'train':
    print(f">> Training on {train_files}, testing on {devel_files}")
elif MODE == 'test':
    print(f">> Loading an already trained model, testing on {devel_files}")
else:
    raise NotImplementedError

print(f">> activation   : {ACT}")
print(f">> layers       : {NUM_LAYERS}")
if NUM_LAYERS == 2:
    print(f">> dimensions   : 768 >> {HIDDEN_DIM} >> num_labels")
elif NUM_LAYERS == 3:
    print(f">> dimensions   : 768 >> {HIDDEN_DIM} >> {HIDDEN_DIM//2} >> num_labels")
print(f">> model        : {MODEL}")
print(f">> first        : {FIRST}")
print(f">> learning rate: {LEARNING_RATE}")
print(f">> rm stopwords : {REMOVE_STOPWORDS}")
print(f">> rm stopwords : {REMOVE_STOPWORDS}")
print(f">> add dummy    : {DUMMY_LABEL}")
print(f">> pooler       : {POOLING} {POOLER_LAYER}")
print(f">> absent tok   : '{ABSENT_TOKEN}'")


if 'roberta' in MODEL:
    tokenizer = RobertaTokenizer.from_pretrained(MODEL)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

if MODE == 'train':
    # datasets builder
    train_dataset = TermPolarityDatasetBERT(train_files,
                                            tokenizer=tokenizer,
                                            absent_token=ABSENT_TOKEN,
                                            stopwords=REMOVE_STOPWORDS,
                                            stopset=stopset,
                                            device=device,
                                            first=FIRST,
                                            roberta=('roberta' in MODEL),
                                            debug=False)

    devel_dataset = TermPolarityDatasetBERT(devel_files,
                                            tokenizer=tokenizer,
                                            absent_token=ABSENT_TOKEN,
                                            stopwords=REMOVE_STOPWORDS,
                                            stopset=stopset,
                                            device=device,
                                            first=FIRST,
                                            roberta=('roberta' in MODEL),
                                            debug=False)

    # hyperparameters
    hparams = { 'vocab'           : train_dataset.vocab,
                'l_vocab'         : train_dataset.l_vocab,
                'l_vocab_itos'    : train_dataset.l_vocab_itos,
                'vocab_size'      : len(train_dataset.vocab),
                'num_classes'     : len(train_dataset.l_vocab), 
                'tokenizer'       : tokenizer,
                'dropout'         : DROPOUT, 
                'learning_rate'   : LEARNING_RATE, 
                'device'          : device,
                'stopset'         : stopset,
                'bert_type'       : MODEL,
                'first'           : FIRST,
                'model'           : MODEL,
                'classifier'      : ACT,
                'layers'          : NUM_LAYERS,
                'hidden_dim'      : HIDDEN_DIM,
                'remove_stopwords': REMOVE_STOPWORDS,
                'dummy'           : DUMMY_LABEL,
                'custom-pooler'   : POOLING,
                'pooler-layer'    : POOLER_LAYER,
                'absent_token'    : ABSENT_TOKEN}
    
    print(f">> num classes: {hparams['num_classes']}")
    
    # model builder
    if 'roberta' in MODEL:
        model = TermPolarityModelROBERTA(hparams)
    else:
        model = TermPolarityModelBERT(hparams)    if DUMMY_LABEL  else    TermPolarityModelBERT_noDummy(hparams)
    

    # callbacks
    early_stop_clbk = clbk.early_stopping.EarlyStopping(monitor='macro_f1',
                                                    patience=5,
                                                    verbose=False,
                                                    mode='max',
                                                    check_on_train_epoch_end=True)

    checkpoint_clbk = clbk.model_checkpoint.ModelCheckpoint(monitor='macro_f1',
                                                mode='max',
                                                save_top_k=1,
                                                save_last=False,
                                                dirpath=f"./model/model_b/{MODEL.split('-')[0]}",
                                                filename=logname+"_{epoch}_{step}_{macro_f1:.2f}")

    logger  = pl.loggers.TensorBoardLogger( save_dir=f'./logs/logs_b/{MODEL.split("-")[0]}',
                                            name=logname)

    train_dataloader = DataLoader(train_dataset, 
                            shuffle=True,
                            batch_size=TRAIN_BATCH_SIZE,
                            collate_fn=train_dataset.collate_fn)

    devel_dataloader = DataLoader(devel_dataset, 
                            batch_size=VAL_BATCH_SIZE,
                            collate_fn=devel_dataset.collate_fn)

    
    # train phase
    trainer = pl.Trainer(gpus=1,
                        logger=logger,
                        max_epochs=MAX_EPOCHS,
                        val_check_interval=1.0, 
                        callbacks=[checkpoint_clbk, early_stop_clbk])

    trainer.fit(model, train_dataloader, devel_dataloader)
    best_model_path = checkpoint_clbk.best_model_path

# test best model
if MODE != 'train':

    if 'roberta' in MODEL:
        stoi = {NEUTRAL  : 0,
                POSITIVE : 1,
                NEGATIVE : 2,
                CONFLICT : 3,
                ABSENT   : 4}
        best_model_path = "model/model_b/"+\
            "roberta/roberta-SeqCls_base-1_text-first_epoch=8_step=1997_macro_f1=54.49.ckpt"
            #"ROBERTA-SeqCls_epoch=4_step=1109_macro_f1=53.69.ckpt"
    else:
        if DUMMY_LABEL:
            stoi = {NEUTRAL  : 0,
                    POSITIVE : 1,
                    NEGATIVE : 2,
                    CONFLICT : 3,
                    ABSENT   : 4}
            best_model_path = "model/model_b/"+\
                "bert/bert-SeqCls_gelu-2_term-first_concat-max[-2]_epoch=4_step=1109_macro_f1=50.54.ckpt"
                #"BERT-SeqCls-model_gelu-classifier-3_both-to-both_term_epoch=8_step=1997_macro_f1=53.72-BEST.ckpt"
        else:
            stoi = {NEUTRAL  : 0,
                    POSITIVE : 1,
                    NEGATIVE : 2,
                    CONFLICT : 3}
            best_model_path = "model/model_b/"+"BERT-SeqCls-noDummy_selu-2_uncased_term_epoch=14_step=2324_macro_f1=54.41.ckpt"
    
    itos = {v : k for k, v in stoi.items()}

    hparams = { 'l_vocab'         : stoi,
                'l_vocab_itos'    : itos,
                'num_classes'     : len(stoi),
                'tokenizer'       : tokenizer,
                'dropout'         : DROPOUT, 
                'learning_rate'   : LEARNING_RATE, 
                'device'          : device,
                'stopset'         : stopset,
                'bert_type'       : MODEL,
                'first'           : FIRST,
                'model'           : MODEL,
                'classifier'      : ACT,
                'layers'          : NUM_LAYERS,
                'hidden_dim'      : HIDDEN_DIM,
                'remove_stopwords': REMOVE_STOPWORDS,
                'dummy'           : DUMMY_LABEL,
                'custom-pooler'   : POOLING,
                'pooler-layer'    : POOLER_LAYER,
                'absent_token'    : ABSENT_TOKEN}


print(f">> Testing best model -- path: {best_model_path}\n\n")

if torch.cuda.is_available():
    if 'roberta' in MODEL:
        model = TermPolarityModelROBERTA.load_from_checkpoint(best_model_path, hparams=hparams).cuda()
    else:
        if DUMMY_LABEL:
            model = TermPolarityModelBERT.load_from_checkpoint(best_model_path, hparams=hparams).cuda()
        else:
            model = TermPolarityModelBERT_noDummy.load_from_checkpoint(best_model_path, hparams=hparams).cuda()
else:
    if 'roberta' in MODEL:
        model = TermPolarityModelROBERTA.load_from_checkpoint(best_model_path, hparams=hparams)
    else:
        if DUMMY_LABEL:
            model = TermPolarityModelBERT.load_from_checkpoint(best_model_path, hparams=hparams)
        else:
            model = TermPolarityModelBERT_noDummy.load_from_checkpoint(best_model_path, hparams=hparams)

if DUMMY_LABEL:
    test_dataset = TermPolarityDatasetBERT.test_dataset_builder(devel_files) # returns a List[Dict] as in the docker file
else:
    test_dataset = TermPolarityDatasetBERT_noDummy.test_dataset_builder(devel_files) # returns a List[Dict] as in the docker file
 
preds_task_b = model.predict_task_b(raw_data=test_dataset, l_vocab=hparams['l_vocab'], device=device)
model.sanitize_labels(test_dataset, absent_token=ABSENT_TOKEN)
print("len(data)   =", len(test_dataset))
print("len(result) =", len(preds_task_b))
assert len(test_dataset) == len(preds_task_b)

if MODE == 'train':
    print(f"\n\n## {logname}", file=open("/media/nemo/DATA/uni/nlp-hw2/hw2/stud/results/results_B.txt", 'a'))
eval_task_B(test_dataset, preds_task_b, display=True, tofile=(MODE == 'train'))
