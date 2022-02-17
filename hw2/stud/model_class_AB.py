import torch
import pytorch_lightning as pl
try:
    from utils          import *
    from model_class_A  import TermIdentificationModelBERT
    from model_class_B  import TermPolarityModelBERT, TermPolarityModelBERT_noDummy, TermPolarityModelROBERTA
except:
    from stud.utils          import *
    from stud.model_class_A  import TermIdentificationModelBERT
    from stud.model_class_B  import TermPolarityModelBERT, TermPolarityModelBERT_noDummy, TermPolarityModelROBERTA
 

class ModelAB(pl.LightningModule):
    def __init__(self, hparams_a, hparams_b):
        super(ModelAB, self).__init__()
        self.hparams_a = hparams_a
        self.hparams_b = hparams_b
        
        if torch.cuda.is_available():
            self.model_a = TermIdentificationModelBERT.load_from_checkpoint(hparams_a['best_model_path'], hparams=hparams_a).cuda()
            if 'roberta' in hparams_b['bert_type']:
                self.model_b = TermPolarityModelROBERTA.load_from_checkpoint(hparams_b['best_model_path'], hparams=hparams_b).cuda()
            else:
                self.model_b = TermPolarityModelBERT.load_from_checkpoint(hparams_b['best_model_path'], hparams=hparams_b).cuda()
        else:
            self.model_a = TermIdentificationModelBERT.load_from_checkpoint(hparams_a['best_model_path'], hparams=hparams_a)            
            if 'roberta' in hparams_b['bert_type']:
                self.model_b = TermPolarityModelROBERTA.load_from_checkpoint(hparams_b['best_model_path'], hparams=hparams_b)
            else:
                self.model_b = TermPolarityModelBERT.load_from_checkpoint(hparams_b['best_model_path'], hparams=hparams_b)
    

    def predict_task_a(self, data: List[Dict]):
        """
        Executes the predict for task A (Aspect Term Identification)
        """
        return self.model_a.predict_task_a(data=data)


    def predict_task_b(self, raw_data):
        """
        Executes the predict for task B (Aspect Term Polarity)
        """
        result = self.model_b.predict_task_b(raw_data=raw_data, l_vocab=self.hparams_b['l_vocab'])
        self.model_b.sanitize_labels(raw_data, absent_token=self.hparams_b['absent_token'])
        return result
    

    def predict_task_ab(self, data_task_a):
        preds_a = self.predict_task_a(data=data_task_a)
        data_task_b = []
        for elem, pred in zip(data_task_a, preds_a):
            þ = []
            for p in pred['targets']:
                þ.append([[0,0], p[0], 'unlabeled'])
            data_task_b.append({'text'    : elem['text'],
                                'targets' : þ})
        preds_b = self.predict_task_b(raw_data=data_task_b)
        return preds_b






class ModelAB_noDummy(ModelAB):
    def __init__(self, hparams_a, hparams_b):
        super().__init__(hparams_a, hparams_b)
        self.hparams_a = hparams_a
        self.hparams_b = hparams_b
        if torch.cuda.is_available():
            self.model_a = TermIdentificationModelBERT.load_from_checkpoint(hparams_a['best_model_path'], hparams=hparams_a).cuda()
            self.model_b = TermPolarityModelBERT_noDummy.load_from_checkpoint(hparams_b['best_model_path'], hparams=hparams_b).cuda()
        else:
            self.model_a = TermIdentificationModelBERT.load_from_checkpoint(hparams_a['best_model_path'], hparams=hparams_a)
            self.model_b = TermPolarityModelBERT_noDummy.load_from_checkpoint(hparams_b['best_model_path'], hparams=hparams_b)