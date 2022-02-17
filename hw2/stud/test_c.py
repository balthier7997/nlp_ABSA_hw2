import gc
import nltk
import torch
import argparse
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning import callbacks as clbk

from transformers import AutoTokenizer

from utils               import *
from model_class_C       import CategoryExtractionMultiLabelModelBERT, CategoryExtractionMultiLabelModelROBERTA
from dataset_class_extra import CategoryExtractionDatasetBERT
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
MODEL            = args.model
DROPOUT          = args.dropout
NUM_LAYERS       = args.num_layers
HIDDEN_DIM       = args.hidden_dim
LEARNING_RATE    = args.learning_rate
REMOVE_STOPWORDS = args.rm_stopwords


MAX_EPOCHS = 15
TRAIN_BATCH_SIZE = 15       #NOTE: MUST BE A MULTIPLE OF 5!!!
VAL_BATCH_SIZE   = 35       #NOTE: MUST BE A MULTIPLE OF 5!!!
train_files = [TRAIN_FILE_R]
devel_files = [DEVEL_FILE_R]

logname = f"C_{MODEL.split('-')[0]}-MultiLabel_{ACT}-{NUM_LAYERS}_lr={LEARNING_RATE}_drop={DROPOUT}"


if REMOVE_STOPWORDS:
    nltk.download('stopwords', download_dir='./nltk_data')
    nltk.data.path.append('./nltk_data')
    stopset = set(nltk.corpus.stopwords.words('english'))
    logname += f'_rm-stops'
else:
    stopset = None

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
print(f">> dropout      : {DROPOUT}")
print(f">> learning rate: {LEARNING_RATE}")
print(f">> rm stopwords : {REMOVE_STOPWORDS}")


tokenizer = AutoTokenizer.from_pretrained(MODEL)

if MODE == 'train':
    # datasets builder
    train_dataset = CategoryExtractionDatasetBERT(train_files, tokenizer=tokenizer, roberta=('roberta' in MODEL), device=device, debug=False)
    devel_dataset = CategoryExtractionDatasetBERT(devel_files, tokenizer=tokenizer, roberta=('roberta' in MODEL), device=device, debug=False)

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
                'classifier'      : ACT,
                'bert_type'       : MODEL,
                'model'           : MODEL,
                'hidden_dim'      : HIDDEN_DIM,
                'layers'          : NUM_LAYERS,
                'remove_stopwords': REMOVE_STOPWORDS,
                'uncased'         : ('uncased' in MODEL)}
    print(f">> Num classes: {hparams['num_classes']}")
    

    # model builder
    if 'roberta' in MODEL:
        model = CategoryExtractionMultiLabelModelROBERTA(hparams)
    else:
        model = CategoryExtractionMultiLabelModelBERT(hparams)
    

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
                                                dirpath=f"./model/model_c/{MODEL.split('-')[0]}",
                                                filename=logname+"_{epoch}_{step}_{macro_f1:.2f}")

    logger  = pl.loggers.TensorBoardLogger( save_dir=f'./logs/logs_c/multilabel/{MODEL.split("-")[0]}/',
                                            name=logname, 
                                            version=None)

    train_dataloader = DataLoader(train_dataset, 
                                shuffle=False,                          #NOTE: MUST BE FALSE!!!
                                batch_size=TRAIN_BATCH_SIZE,
                                collate_fn=train_dataset.collate_fn_wrapper)

    devel_dataloader = DataLoader(devel_dataset, 
                                shuffle=False,                          #NOTE: MUST BE FALSE!!!
                                batch_size=VAL_BATCH_SIZE,
                                collate_fn=devel_dataset.collate_fn_wrapper)

    
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
    stoi = {FOOD     : 0,
            MISC     : 1,
            SERVICE  : 2,
            AMBIENCE : 3,
            PRICE    : 4}
    itos = {v : k for k, v in stoi.items()}

    best_model_path = "model/model_c/roberta/"+\
        "C_roberta-MultiLabel_selu-3_epoch=2_step=500_macro_f1=86.06.ckpt"
    hparams = { 'bert_type'       : MODEL,
                'num_classes'     : len(stoi),
                'l_vocab'         : stoi, 
                'l_vocab_itos'    : itos,
                'tokenizer'       : tokenizer,
                'dropout'         : DROPOUT, 
                'hidden_dim'      : HIDDEN_DIM,
                'learning_rate'   : LEARNING_RATE,
                'classifier'      : ACT,
                'device'          : device,
                'layers'          : NUM_LAYERS}

print(">> Testing best model -- path:", best_model_path)
if torch.cuda.is_available():
    if 'roberta' in MODEL:
        model = CategoryExtractionMultiLabelModelROBERTA(hparams).load_from_checkpoint(best_model_path, hparams=hparams).cuda()
    else:
        model = CategoryExtractionMultiLabelModelBERT.load_from_checkpoint(best_model_path, hparams=hparams).cuda()
else:
    if 'roberta' in MODEL:
        model = CategoryExtractionMultiLabelModelROBERTA(hparams).load_from_checkpoint(best_model_path, hparams=hparams)
    else:
        model = CategoryExtractionMultiLabelModelBERT.load_from_checkpoint(best_model_path, hparams=hparams)

test_dataset = CategoryExtractionDatasetBERT.test_dataset_builder(devel_files) # returns a List[Dict] as in the docker file
 
preds_task_c = model.predict_task_c(raw_data=test_dataset, l_vocab=hparams['l_vocab'], device=device)
if MODE == 'train':
    print(f"\n\n## {logname}", file=open("/media/nemo/DATA/uni/nlp-hw2/hw2/stud/results/results_C.txt", 'a'))
eval_task_C(test_dataset, preds_task_c, display=True, tofile=True if MODE == 'train' else False)
