import torch
import pytorch_lightning as pl

try:
    from utils          import *
    from model_class_C  import CategoryExtractionMultiLabelModelBERT, CategoryExtractionMultiLabelModelROBERTA
    from model_class_D  import CategorySentimentModelBERT, CategorySentimentModelROBERTA
except:
    from stud.utils          import *
    from stud.model_class_C  import CategoryExtractionMultiLabelModelBERT, CategoryExtractionMultiLabelModelROBERTA
    from stud.model_class_D  import CategorySentimentModelBERT, CategorySentimentModelROBERTA


class ModelCD(pl.LightningModule):
    def __init__(self, hparams_c, hparams_d):
        super(ModelCD, self).__init__()
        self.hparams_c = hparams_c
        self.hparams_d = hparams_d
        
        if torch.cuda.is_available():
            if 'roberta' in hparams_c['bert_type']:
                self.model_c = CategoryExtractionMultiLabelModelROBERTA.load_from_checkpoint(   hparams_c['best_model_path'],
                                                                                                hparams=hparams_c).cuda()
            else:
                self.model_c = CategoryExtractionMultiLabelModelBERT.load_from_checkpoint(  hparams_c['best_model_path'],
                                                                                            hparams=hparams_c).cuda()
            
            if 'roberta' in hparams_c['bert_type']:
                self.model_d = CategorySentimentModelROBERTA.load_from_checkpoint(  hparams_d['best_model_path'],
                                                                                    hparams=hparams_d).cuda()
            else:
                self.model_d = CategorySentimentModelBERT.load_from_checkpoint( hparams_d['best_model_path'],
                                                                            hparams=hparams_d).cuda()
        else:
            if 'roberta' in hparams_c['bert_type']:
                self.model_c = CategoryExtractionMultiLabelModelROBERTA.load_from_checkpoint(   hparams_c['best_model_path'],
                                                                                                hparams=hparams_c)
            else:
                self.model_c = CategoryExtractionMultiLabelModelBERT.load_from_checkpoint(  hparams_c['best_model_path'], 
                                                                                            hparams=hparams_c)
            
            if 'roberta' in hparams_c['bert_type']:
                self.model_d = CategorySentimentModelROBERTA.load_from_checkpoint(  hparams_d['best_model_path'],
                                                                                    hparams=hparams_d)
            else:
                self.model_d = CategorySentimentModelBERT.load_from_checkpoint( hparams_d['best_model_path'],
                                                                                hparams=hparams_d)
    

    def predict_task_c(self, data: List[Dict]):
        """
        Executes the predict for task C (Category Extraction)
        """
        return self.model_c.predict_task_c(raw_data=data, l_vocab=self.hparams_c['l_vocab'], device=self.device)


    def predict_task_d(self, raw_data):
        """
        Executes the predict for task D (Category Sentiment)
        """
        result = self.model_d.predict_task_d(raw_data=raw_data, l_vocab=self.hparams_d['l_vocab'], device=self.device)
        return result
    

    def predict_task_cd(self, data_task_c):
        preds_c = self.predict_task_c(data_task_c)
        data_task_d = []
        assert len(data_task_c) == len(preds_c)
        for elem, pred in zip(data_task_c, preds_c):
            data_task_d.append({'text'       : elem['text'],
                                'categories' : pred['categories']})
        return self.predict_task_d(data_task_d)
