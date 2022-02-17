import gc
import nltk
import torch
import argparse
import pytorch_lightning as pl

from torch.utils.data  import DataLoader
from pytorch_lightning import callbacks as clbk

from transformers import AutoTokenizer, RobertaTokenizer

from utils          import *
from model_class_A  import PreTrainedEmbeddingLayer,  TermIdentificationModel, TermIdentificationModelBERT
from dataset_class  import TermIdentificationDataset, TermIdentificationDatasetBERT
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
                    default=5e-5)
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

MAX_EPOCHS       = 15
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE   = 64
train_files = [TRAIN_FILE_R, TRAIN_FILE_L]
devel_files = [DEVEL_FILE_R, DEVEL_FILE_L]

logname = f"A_{MODEL.split('-')[0]}-TokCls_{ACT}-{NUM_LAYERS}_lr={LEARNING_RATE}_drop={DROPOUT}_hidden-dim={HIDDEN_DIM}"


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


if 'roberta' in MODEL:
    tokenizer = RobertaTokenizer.from_pretrained(MODEL)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

if 'bert' in MODEL:
    # dataset loader
    train_dataset = TermIdentificationDatasetBERT(  train_files,
                                                    tokenizer=tokenizer, 
                                                    rm_stopwords=REMOVE_STOPWORDS,
                                                    stopset=stopset,
                                                    roberta=('roberta' in MODEL))

    devel_dataset = TermIdentificationDatasetBERT(  devel_files,
                                                    tokenizer=tokenizer, 
                                                    rm_stopwords=REMOVE_STOPWORDS,
                                                    stopset=stopset,
                                                    roberta=('roberta' in MODEL))

    # hyperparameters
    hparams = { 'vocab'           : train_dataset.vocab,
                'l_vocab'         : train_dataset.l_vocab,
                'l_vocab_itos'    : train_dataset.l_vocab_itos,
                'vocab_size'      : len(train_dataset.vocab),
                'num_classes'     : len(train_dataset.l_vocab),
                'cls_token'       : train_dataset.cls_token,
                'sep_token'       : train_dataset.sep_token,
                'unk_token'       : train_dataset.unk_token,
                'pad_token'       : train_dataset.pad_token,
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
    
    
    if MODE == 'train':
        # model builder
        model = TermIdentificationModelBERT(hparams)
        
        # callbacks
        early_stop_clbk = clbk.early_stopping.EarlyStopping(monitor='F1_score',
                                                        patience=5,
                                                        verbose=False,
                                                        mode='max',
                                                        check_on_train_epoch_end=True)

        checkpoint_clbk = clbk.model_checkpoint.ModelCheckpoint(monitor='F1_score',
                                                    mode='max',
                                                    save_top_k=1,
                                                    save_last=False,
                                                    dirpath=f"./model/model_a/{MODEL.split('-')[0]}",
                                                    filename=logname+"_{epoch}_{step}_{F1_score:.2f}")

        logger  = pl.loggers.TensorBoardLogger( save_dir=f'./logs/logs_a/{MODEL.split("-")[0]}',
                                                name=logname)
        
        train_dataloader = DataLoader(train_dataset, 
                                    shuffle=True,
                                    batch_size=TRAIN_BATCH_SIZE,
                                    collate_fn=train_dataset.collate_fn_BERT)#,
                                    #num_workers=12)

        devel_dataloader = DataLoader(devel_dataset, 
                                    batch_size=VAL_BATCH_SIZE,
                                    collate_fn=devel_dataset.collate_fn_BERT)
  
        
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
        #best_model_path = "model/model_a/BERT-model_both-to-both_optimized_relu-2_epoch=5_step=941_train_loss=0.00_F1_score=80.19.ckpt"
        #best_model_path = "model/model_a/BERT_uncased_optimized_relu-2_epoch=12_step=2040_train_loss=0.02_F1_score=69.19.ckpt"
        best_model_path = "model/model_a/roberta/roberta-TokCls_tanh-3_epoch=6_step=1098_F1_score=83.10.ckpt"
    
    print(">> Testing best model -- path:", best_model_path, '\n\n')
    if torch.cuda.is_available():
        model = TermIdentificationModelBERT(hparams).load_from_checkpoint(best_model_path, hparams=hparams).cuda()
    else:
        model = TermIdentificationModelBERT(hparams).load_from_checkpoint(best_model_path, hparams=hparams)


    model.eval()
    preds_task_a = model.predict_task_a(devel_dataset)
    if MODE == 'train':
        print(f"\n\n## {logname}", file=open("/media/nemo/DATA/uni/nlp-hw2/hw2/stud/results/results_A.txt", 'a'))
    eval_task_A(devel_dataset, preds_task_a, display=True, debug=True, tofile=(MODE == 'train'))
 
    
else:
    train_dataset = TermIdentificationDataset(train_files, size=-1, vocab=None)
    devel_dataset = TermIdentificationDataset(devel_files, size=-1, vocab=train_dataset.vocab)

    # hyperparameters
    hparams = { 'vocab'           : train_dataset.vocab,
                'l_vocab'         : train_dataset.l_vocab,
                'l_vocab_itos'    : train_dataset.l_vocab_itos,
                'vocab_size'      : len(train_dataset.vocab),
                'pad_token'       : '<pad>',
                # 'glove', 'domain-restaurant', 'domain-laptop', 'glove+rest_domain'
                # 'glove+lapt_domain', 'glove+all_domains', 'all_domains'
                'embedding_type'  : 'glove',#+all_domains',
                
                # 50d, 100d, 200d, 300d
                'embedding_dim'   : 200,
                
                # 6B, 42B, 840B
                'B'               : 6,
                'lstm_hidden_dim' : 128,
                'lstm_bidirect'   : True, 
                'lstm_layers'     : 2, 
                'num_classes'     : len(train_dataset.l_vocab), 
                'dropout'         : 0.0, 
                'device'          : device,
                'model'           : MODEL}

    # retrieve embeddings
    emb = PreTrainedEmbeddingLayer(hparams)   # loads pre-trained embeddings

    if MODE == 'train':
        # model builder
        model = TermIdentificationModel(hparams, emb.get_embeddings()).cuda()

        early_stop_clbk = clbk.early_stopping.EarlyStopping(monitor='F1_score',
                                                        patience=15,
                                                        verbose=False,
                                                        mode='max',
                                                        check_on_train_epoch_end=True)

        logname = f"{hparams['model']}-model_{hparams['embedding_type']}_{hparams['B']}B_{hparams['embedding_dim']}d_{MODE}"
        checkpoint_clbk = clbk.model_checkpoint.ModelCheckpoint(monitor='F1_score',
                                                    mode='max',
                                                    save_top_k=1,
                                                    save_last=False,
                                                    dirpath="./model/model_a",
                                                    filename=logname+"_{epoch}_{step}_{train_loss:.4f}_{F1_score:.4f}")

        # train phase
        logger  = pl.loggers.TensorBoardLogger(save_dir='./logs/logs_a',
                                            name=logname)

        train_dataloader = DataLoader(train_dataset, 
                                    batch_size=TRAIN_BATCH_SIZE,
                                    collate_fn=TermIdentificationDataset.collate_fn)

        devel_dataloader = DataLoader(devel_dataset, 
                                    batch_size=VAL_BATCH_SIZE,
                                    collate_fn=TermIdentificationDataset.collate_fn)

        trainer = pl.Trainer(gpus=1,
                            logger=logger,
                            max_epochs=MAX_EPOCHS,
                            val_check_interval=1.0,
                            callbacks=[checkpoint_clbk, early_stop_clbk])

        trainer.fit(model, train_dataloader, devel_dataloader)
        best_model_path = checkpoint_clbk.best_model_path

    # test best model
    if MODE != 'train':
        best_model_path = "model/BASE-model_200d_both-to-both_epoch=7_step=1255_train_loss=0.0164_F1_score=56.0008.ckpt"

    
    print(">> Testing best model -- path:", best_model_path)
    model = TermIdentificationModel.load_from_checkpoint(best_model_path,
                                                        hparams=hparams,
                                                        embedding=emb.get_embeddings()).cpu()


    preds_a = model.predict_task_a(devel_dataset)
    eval_task_A(devel_dataset, preds_a, display=True, debug=False)



    label_vocab_itos = train_dataset.l_vocab_itos

    f1_scores = compute_f1_seqtag_score(model, devel_dataset)
    print_seqtag_results(f1_scores['macro_f1'], f1_scores['micro_f1'], f1_scores["class_f1"])