import gc
import nltk
import torch
import argparse
import pytorch_lightning as pl

from torch.utils.data   import DataLoader
from transformers       import AutoTokenizer
from pytorch_lightning  import callbacks as clbk

from utils               import *
from model_class_D       import CategorySentimentModelBERT, CategorySentimentModelROBERTA
from dataset_class_extra import CategorySentimentDatasetBERT
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
                    help="category or text", type=str, default='category')
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
DROPOUT          = args.dropout
POOLING          = args.pooling
HIDDEN_DIM       = args.hidden_dim
NUM_LAYERS       = args.num_layers
POOLER_LAYER     = [elem for elem in args.pooling_layer] if args.pooling_layer is not None else args.pooling_layer
LEARNING_RATE    = args.learning_rate
REMOVE_STOPWORDS = args.rm_stopwords



MAX_EPOCHS = 30
TRAIN_BATCH_SIZE = 35
VAL_BATCH_SIZE   = 35
train_files = [TRAIN_FILE_R]
devel_files = [DEVEL_FILE_R]

logname = f"D_{MODEL.split('-')[0]}-SeqCls_{ACT}-{NUM_LAYERS}_{FIRST}-first_lr={LEARNING_RATE}_drop={DROPOUT}"


if POOLING is not None:
    logname += f'_{POOLING}{POOLER_LAYER}'

if REMOVE_STOPWORDS:
    nltk.download('stopwords', download_dir='./nltk_data')
    nltk.data.path.append('./nltk_data')
    stopset = set(nltk.corpus.stopwords.words('english'))
    logname += f'_rm-stops'
else:
    stopset = {}


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
print(f">> dropout      : {DROPOUT}")
print(f">> learning rate: {LEARNING_RATE}")
print(f">> rm stopwords : {REMOVE_STOPWORDS}")
print(f">> pooler       : {POOLING} {POOLER_LAYER}")

tokenizer = AutoTokenizer.from_pretrained(MODEL)

if MODE == 'train':
    # datasets builder
    train_dataset = CategorySentimentDatasetBERT(train_files,
                                                first=FIRST, 
                                                device=device,
                                                stopset=stopset, 
                                                tokenizer=tokenizer, 
                                                stopwords=REMOVE_STOPWORDS,
                                                roberta=('roberta' in MODEL),
                                                debug=False)

    devel_dataset = CategorySentimentDatasetBERT(devel_files,
                                                first=FIRST, 
                                                device=device,
                                                stopset=stopset, 
                                                tokenizer=tokenizer, 
                                                stopwords=REMOVE_STOPWORDS,
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
                'hidden_dim'      : HIDDEN_DIM,
                'layers'          : NUM_LAYERS,
                'remove_stopwords': REMOVE_STOPWORDS,
                'custom-pooler'   : POOLING,
                'pooler-layer'    : POOLER_LAYER}
    print(f">> Num classes: {hparams['num_classes']}")
    

    # model builder
    if 'roberta' in MODEL:
        model = CategorySentimentModelROBERTA(hparams)
    else:
        model = CategorySentimentModelBERT(hparams)


    # callbacks
    early_stop_clbk = clbk.early_stopping.EarlyStopping(monitor='macro_f1',
                                                    patience=5,
                                                    verbose=True,
                                                    mode='max',
                                                    check_on_train_epoch_end=True)

    checkpoint_clbk = clbk.model_checkpoint.ModelCheckpoint(monitor='macro_f1',
                                                mode='max',
                                                save_top_k=1,
                                                save_last=False,
                                                dirpath="./model/model_d/roberta/",
                                                filename=logname+"_{epoch}_{step}_{macro_f1:.2f}")

    #logger = pl.loggers.WandbLogger()
    logger  = pl.loggers.TensorBoardLogger( save_dir=f'./logs/logs_d/{MODEL.split("-")[0]}', name=logname)

    train_dataloader = DataLoader(train_dataset, 
                            shuffle=False,
                            batch_size=TRAIN_BATCH_SIZE,
                            collate_fn=train_dataset.collate_fn_wrapper)

    devel_dataloader = DataLoader(devel_dataset, 
                            shuffle=False,
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
    stoi = {NEUTRAL  : 0,
            POSITIVE : 1,
            NEGATIVE : 2,
            CONFLICT : 3}
    itos = {v : k for k, v in stoi.items()}

    best_model_path = "model/model_d/"+\
        "BERT-SeqCls-uncased_selu-cls-2_category-first_epoch=10_step=967_macro_f1=68.59-BEST.ckpt"
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
                'hidden_dim'      : HIDDEN_DIM,
                'layers'          : NUM_LAYERS,
                'remove_stopwords': REMOVE_STOPWORDS,
                'custom-pooler'   : POOLING,
                'pooler-layer'    : POOLER_LAYER}

print(">> Testing best model -- path:", best_model_path)
if torch.cuda.is_available():
    if 'roberta' in MODEL:
        model = CategorySentimentModelROBERTA.load_from_checkpoint(best_model_path, hparams=hparams).cuda()
    else:
        model = CategorySentimentModelBERT.load_from_checkpoint(best_model_path, hparams=hparams).cuda()
else:
    if 'roberta' in MODEL:
        model = CategorySentimentModelROBERTA.load_from_checkpoint(best_model_path, hparams=hparams)
    else:
        model = CategorySentimentModelBERT.load_from_checkpoint(best_model_path, hparams=hparams)

test_dataset = CategorySentimentDatasetBERT.test_dataset_builder(devel_files) # returns a List[Dict] as in the docker file
 
preds_task_c = model.predict_task_d(raw_data=test_dataset, l_vocab=hparams['l_vocab'], device=device)
if MODE == 'train':
    print(f"\n\n## {logname}", file=open("/media/nemo/DATA/uni/nlp-hw2/hw2/stud/results/results_D.txt", 'a'))
eval_task_D(test_dataset, preds_task_c, display=True, tofile=(MODE == 'train'))
