# Import libraries
import torch, torchmetrics, timm, wandb, pytorch_lightning as pl, os
from torch import nn
from torchmetrics import F1Score, Precision, Accuracy
from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback
from time import time
from utils import get_fm
from loss import ContrastiveLoss

class TripletModel(pl.LightningModule):
    
    """"
    
    This class gets several arguments and returns a model for training.
    
    Parameters:
    
        input_shape  - shape of input to the model, tuple -> int;
        model_name   - name of the model from timm library, str;
        num_classes  - number of classes to be outputed from the model, int;
        lr           - learning rate value, float.
    
    """
    
    def __init__(self, input_shape, model_name, num_classes, ds_name, margin, lr = 2e-4):
        super().__init__()
        
        # Log hyperparameters
        self.save_hyperparameters()
        self.lr, self.ds_name = lr, ds_name
        # Evaluation metric
        self.f1 = F1Score(task = "multiclass", num_classes = num_classes)
        self.pr = Precision(task = "multiclass", average='macro', num_classes = num_classes)
        self.accuracy = Accuracy(task = "multiclass", num_classes = num_classes)
        
        self.cos_loss = torch.nn.CosineEmbeddingLoss(margin = margin)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.con_loss = ContrastiveLoss(margin)
        
        self.cos = torch.nn.CosineSimilarity(dim = 1, eps = 1e-6)
        
        self.lbls = {"cos_pos": torch.tensor(1.).unsqueeze(0), "cos_neg": torch.tensor(-1.).unsqueeze(0),
                     "con_pos": torch.tensor(1.).unsqueeze(0), "con_neg": torch.tensor(0.).unsqueeze(0)}
        
        # Get model to be trained
        self.model = timm.create_model(model_name, pretrained = True, num_classes = num_classes)
        self.train_times, self.validation_times = [], []

    # Get optimizere to update trainable parameters
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr = self.lr)
        
    # Feed forward of the model
    def forward(self, inp): return self.model(inp)
    
    # Set time when the epoch is started
    def on_train_epoch_start(self): self.train_start_time = time()
    
    # Compute time when the epoch is finished
    def on_train_epoch_end(self): self.train_elapsed_time = time() - self.train_start_time; self.train_times.append(self.train_elapsed_time); self.log("train_time", self.train_elapsed_time, prog_bar = True)
        
    def training_step(self, batch, batch_idx):
        
        """
        
        This function gets several parameters and conducts training step for a single batch.
        
        Parameters:
        
            batch      - a single batch of the dataloader, batch object;
            batch_idx  - index of the abovementioned batch, int.
            
        Output:
        
            loss       - loss value for the particular mini-batch with images, tensor.
            
        """
        
        # Get images and their corresponding labels
        if "sketch" in self.ds_name:
            
            qry_ims, pos_ims, neg_ims, qry_im_lbls, pos_im_lbls, neg_im_lbls = batch["qry_im"], batch["pos_im"], batch["neg_im"], batch["qry_im_lbl"], batch["pos_im_lbl"], batch["neg_im_lbl"]
            
            qry_fms = self.model.forward_features(qry_ims)
            pos_fms = self.model.forward_features(pos_ims)
            neg_fms = self.model.forward_features(neg_ims)

            pred_qry_lbls = self.model.forward_head(qry_fms)
            pred_pos_lbls = self.model.forward_head(pos_fms)
            pred_neg_lbls = self.model.forward_head(neg_fms)
            
            qry_fms, pos_fms, neg_fms = get_fm(qry_fms), get_fm(pos_fms), get_fm(neg_fms)
        
            ce_qry_loss = self.ce_loss(pred_qry_lbls, qry_im_lbls)
            ce_poss_loss = self.ce_loss(pred_pos_lbls, pos_im_lbls)

            loss = ce_qry_loss + ce_poss_loss
            
            if "cos" in self.ds_name:
                cos_pos_loss = self.cos_loss(qry_fms, pos_fms, self.lbls["cos_pos"].to("cuda"))
                cos_neg_loss = self.cos_loss(qry_fms, neg_fms, self.lbls["cos_neg"].to("cuda"))
                cos_loss = cos_pos_loss + cos_neg_loss
                loss += cos_loss
            
            if "con" in self.ds_name:
                
                con_pos_loss = self.con_loss(qry_fms, pos_fms, self.lbls["con_pos"].to("cuda"))
                con_neg_loss = self.con_loss(qry_fms, neg_fms, self.lbls["con_neg"].to("cuda"))
                con_loss = con_pos_loss + con_neg_loss
                loss += con_loss

            # Train metrics
            qry_lbls = torch.argmax(pred_qry_lbls, dim = 1)
            acc = self.accuracy(qry_lbls, qry_im_lbls)
            pr = self.pr(qry_lbls, qry_im_lbls)
            f1 = self.f1(qry_lbls, qry_im_lbls)
            
            self.log("train_loss", loss, on_step = False, on_epoch = True, logger = True, sync_dist = True)
            self.log("train_acc", acc, on_step = False, on_epoch = True, logger = True, sync_dist = True)
            self.log("train_f1", f1, on_step = False, on_epoch = True, logger = True, sync_dist = True)
            self.log("train_pr", pr, on_step = False, on_epoch = True, logger = True, sync_dist = True)
            
        elif "default" in self.ds_name:
            
            qry_ims, pos_ims, neg_ims, qry_im_lbls, pos_im_lbls, neg_im_lbls = batch["qry_im"], batch["pos_im"], batch["neg_im"], batch["qry_im_lbl"], batch["pos_im_lbl"], batch["neg_im_lbl"]
            
            qry_fms = self.model.forward_features(qry_ims)
            pos_fms = self.model.forward_features(pos_ims)
            neg_fms = self.model.forward_features(neg_ims)

            pred_qry_lbls = self.model.forward_head(qry_fms)
            pred_pos_lbls = self.model.forward_head(pos_fms)
            pred_neg_lbls = self.model.forward_head(neg_fms)
            
            qry_fms, pos_fms, neg_fms = get_fm(qry_fms), get_fm(pos_fms), get_fm(neg_fms)
        
            cos_pos_loss = self.cos_loss(qry_fms, pos_fms, self.lbls["cos_pos"].to("cuda"))
            cos_neg_loss = self.cos_loss(qry_fms, neg_fms, self.lbls["cos_neg"].to("cuda"))
            cos_loss = cos_pos_loss + cos_neg_loss

#             ce_qry_loss = self.ce_loss(pred_qry_lbls, qry_im_lbls)
#             ce_poss_loss = self.ce_loss(pred_pos_lbls, pos_im_lbls)

#             ce_loss = ce_qry_loss + ce_poss_loss

            # loss = cos_loss + ce_loss
            loss = cos_loss

            # Train metrics
            qry_lbls = torch.argmax(pred_qry_lbls, dim = 1)
            acc = self.accuracy(qry_lbls, qry_im_lbls)
            pr = self.pr(qry_lbls, qry_im_lbls)
            f1 = self.f1(qry_lbls, qry_im_lbls)
            
            self.log("train_loss", loss, on_step = False, on_epoch = True, logger = True, sync_dist = True)
            # self.log("train_acc", acc, on_step = False, on_epoch = True, logger = True, sync_dist = True)
            # self.log("train_f1", f1, on_step = False, on_epoch = True, logger = True, sync_dist = True)
            # self.log("train_pr", pr, on_step = False, on_epoch = True, logger = True, sync_dist = True)
            
        else:
            
            x, y = batch
            # Get logits
            logits = self(x)
            # Compute loss        
            loss = self.ce_loss(logits, y)
            # Get indices of the logits with max value
            preds = torch.argmax(logits, dim = 1)
            # Compute accuracy score
            acc = self.accuracy(preds, y)
            # Logs
            self.log("train_loss", loss, on_step = False, on_epoch = True, logger = True); self.log("train_acc", acc, on_step = False, on_epoch = True, logger = True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        """
        
        This function gets several parameters and conducts training step for a single batch.
        
        Parameters:
        
            batch      - a single batch of the dataloader, batch object;
            batch_idx  - index of the abovementioned batch, int.
            
        Output:
        
            loss       - loss value for the particular mini-batch with images, tensor.
            
        """
        
        # Get images and their corresponding labels
        if "sketch" in self.ds_name:
            
            qry_ims, pos_ims, neg_ims, qry_im_lbls, pos_im_lbls, neg_im_lbls = batch["qry_im"], batch["pos_im"], batch["neg_im"], batch["qry_im_lbl"], batch["pos_im_lbl"], batch["neg_im_lbl"]
            
            qry_fms = self.model.forward_features(qry_ims)
            pos_fms = self.model.forward_features(pos_ims)
            neg_fms = self.model.forward_features(neg_ims)

            pred_qry_lbls = self.model.forward_head(qry_fms)
            pred_pos_lbls = self.model.forward_head(pos_fms)
            pred_neg_lbls = self.model.forward_head(neg_fms)
            
            qry_fms, pos_fms, neg_fms = get_fm(qry_fms), get_fm(pos_fms), get_fm(neg_fms)
        
            ce_qry_loss = self.ce_loss(pred_qry_lbls, qry_im_lbls)
            ce_poss_loss = self.ce_loss(pred_pos_lbls, pos_im_lbls)

            loss = ce_qry_loss + ce_poss_loss
            
            if "cos" in self.ds_name:
                cos_pos_loss = self.cos_loss(qry_fms, pos_fms, self.lbls["cos_pos"].to("cuda"))
                cos_neg_loss = self.cos_loss(qry_fms, neg_fms, self.lbls["cos_neg"].to("cuda"))
                cos_loss = cos_pos_loss + cos_neg_loss
                loss += cos_loss
            
            if "con" in self.ds_name:
                
                con_pos_loss = self.con_loss(qry_fms, pos_fms, self.lbls["con_pos"].to("cuda"))
                con_neg_loss = self.con_loss(qry_fms, neg_fms, self.lbls["con_neg"].to("cuda"))
                con_loss = con_pos_loss + con_neg_loss
                loss += con_loss

            # Train metrics
            qry_lbls = torch.argmax(pred_qry_lbls, dim = 1)
            acc = self.accuracy(qry_lbls, qry_im_lbls)
            pr = self.pr(qry_lbls, qry_im_lbls)
            f1 = self.f1(qry_lbls, qry_im_lbls)
            
            top3, top1 = 0, 0
            for idx, lbl_im in enumerate(qry_im_lbls):

                cos_sim = self.cos(qry_fms[idx].unsqueeze(dim = 0), pos_fms)
                try: vals, inds = torch.topk(cos_sim, k = 3)
                except: print(len(cos_sim)); print(len(qry_im_lbls))

                if qry_im_lbls[idx] == qry_im_lbls[inds[0]] or qry_im_lbls[idx] == qry_im_lbls[inds[1]] or qry_im_lbls[idx] == qry_im_lbls[inds[2]]: top3 += 1
                if qry_im_lbls[idx] in qry_im_lbls[inds[0]]: top1 += 1
            
            self.log("valid_loss", loss, on_step = False, on_epoch = True, logger = True, sync_dist = True)
            self.log("valid_acc", acc, on_step = False, on_epoch = True, logger = True, sync_dist = True)
            self.log("valid_f1", f1, on_step = False, on_epoch = True, logger = True, sync_dist = True)
            self.log("valid_pr", pr, on_step = False, on_epoch = True, logger = True, sync_dist = True)
            self.log("top3", top3 / len(qry_im_lbls), on_step = False, on_epoch = True, logger = True, sync_dist = True)
            self.log("top1", top1 / len(qry_im_lbls), on_step = False, on_epoch = True, logger = True, sync_dist = True)
            
        elif "default" in self.ds_name:
            
            qry_ims, pos_ims, neg_ims, qry_im_lbls, pos_im_lbls, neg_im_lbls = batch["qry_im"], batch["pos_im"], batch["neg_im"], batch["qry_im_lbl"], batch["pos_im_lbl"], batch["neg_im_lbl"]
            
            qry_fms = self.model.forward_features(qry_ims)
            pos_fms = self.model.forward_features(pos_ims)
            neg_fms = self.model.forward_features(neg_ims)

            pred_qry_lbls = self.model.forward_head(qry_fms)
            pred_pos_lbls = self.model.forward_head(pos_fms)
            pred_neg_lbls = self.model.forward_head(neg_fms)
            
            qry_fms, pos_fms, neg_fms = get_fm(qry_fms), get_fm(pos_fms), get_fm(neg_fms)
        
            cos_pos_loss = self.cos_loss(qry_fms, pos_fms, self.lbls["cos_pos"].to("cuda"))
            cos_neg_loss = self.cos_loss(qry_fms, neg_fms, self.lbls["cos_neg"].to("cuda"))
            cos_loss = cos_pos_loss + cos_neg_loss

#             ce_qry_loss = self.ce_loss(pred_qry_lbls, qry_im_lbls)
#             ce_poss_loss = self.ce_loss(pred_pos_lbls, pos_im_lbls)

#             ce_loss = ce_qry_loss + ce_poss_loss

            # loss = cos_loss + ce_loss
            loss = cos_loss

            # Train metrics
            qry_lbls = torch.argmax(pred_qry_lbls, dim = 1)
            acc = self.accuracy(qry_lbls, qry_im_lbls)
            pr = self.pr(qry_lbls, qry_im_lbls)
            f1 = self.f1(qry_lbls, qry_im_lbls)
            
            self.log("valid_loss", loss, on_step = False, on_epoch = True, logger = True, sync_dist = True)
            self.log("valid_acc", acc, on_step = False, on_epoch = True, logger = True, sync_dist = True)
            self.log("valid_f1", f1, on_step = False, on_epoch = True, logger = True, sync_dist = True)
            self.log("valid_pr", pr, on_step = False, on_epoch = True, logger = True, sync_dist = True)
            
        else:
            
            x, y = batch
            # Get logits
            logits = self(x)
            # Compute loss        
            loss = self.ce_loss(logits, y)
            # Get indices of the logits with max value
            preds = torch.argmax(logits, dim = 1)
            # Compute accuracy score
            acc = self.accuracy(preds, y)
            # Logs
            self.log("valid_f1_loss", loss, on_step = False, on_epoch = True, logger = True); self.log("valid_f1_acc", acc, on_step = False, on_epoch = True, logger = True)
        
        return loss
    
    # Set the time when validation process is started
    def on_validation_epoch_start(self): self.validation_start_time = time()
    
    # Compute the time when validation process is finished
    def on_validation_epoch_end(self): self.validation_elapsed_time = time() - self.validation_start_time; self.validation_times.append(self.validation_elapsed_time); self.log("valid_time", self.validation_elapsed_time, prog_bar = True)
    
    # Get stats of the train and validation times
    def get_stats(self): return self.train_times, self.validation_times
    
class TripletImagePredictionLogger(Callback):

    """
    
    This class gets several parameters and visualizes several input images and predictions in the end of validation process.

    Parameters:

        val_samples       - validation samples, torch dataloader object;
        cls_names         - class names, list;
        num_samples       - number of samples to be visualized, int.
      
    """
    
    def __init__(self, val_samples, ds_name, cls_names = None, num_samples = 4):
        super().__init__()
        # Get class arguments
        self.num_samples, self.cls_names = num_samples, cls_names
        # Extract images and their corresponding labels
        self.val_imgs = val_samples["qry_im"] if ds_name == "sketch" else val_samples[0]
        self.val_labels = val_samples["qry_im_lbl"] if ds_name == "sketch" else val_samples[1]
        
    def on_validation_epoch_end(self, trainer, pl_module):

        """
        
        This function gets several parameters and visualizes images with their predictions.

        Parameters:

            trainer      - trainer, pytorch lightning trainer object;
            pl_module    - model class, pytorch lightning module object.
        
        """
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        if self.cls_names != None:
            trainer.logger.experiment.log({
                "Sample Validation Prediction Results":[wandb.Image(x, caption=f"Predicted class: {self.cls_names[pred]}, Ground truth class: {self.cls_names[y]}") 
                               for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                     preds[:self.num_samples], 
                                                     val_labels[:self.num_samples])]
                })
            
class Model(pl.LightningModule):
    
    """"
    
    This class gets several arguments and returns a model for training.
    
    Parameters:
    
        input_shape  - shape of input to the model, tuple -> int;
        model_name   - name of the model from timm library, str;
        num_classes  - number of classes to be outputed from the model, int;
        lr           - learning rate value, float.
    
    """
    
    def __init__(self, input_shape, model_name, num_classes, lr = 2e-4):
        super().__init__()
        
        # Log hyperparameters
        self.save_hyperparameters()
        self.lr = lr
        # Evaluation metric
        self.f1 = F1Score(task = "multiclass", num_classes = num_classes)
        self.pr = Precision(task = "multiclass", average='macro', num_classes = num_classes)
        self.accuracy = Accuracy(task = "multiclass", num_classes = num_classes)
        # Get model to be trained
        self.model = timm.create_model(model_name, pretrained = True, num_classes = num_classes)
        self.train_times, self.validation_times = [], []

    # Get optimizere to update trainable parameters
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr = self.lr)
        
    # Feed forward of the model
    def forward(self, inp): return self.model(inp)
    
    # Set time when the epoch is started
    def on_train_epoch_start(self): self.train_start_time = time()
    
    # Compute time when the epoch is finished
    def on_train_epoch_end(self): self.train_elapsed_time = time() - self.train_start_time; self.train_times.append(self.train_elapsed_time); self.log("train_time", self.train_elapsed_time, prog_bar = True)
        
    def training_step(self, batch, batch_idx):
        
        """
        
        This function gets several parameters and conducts training step for a single batch.
        
        Parameters:
        
            batch      - a single batch of the dataloader, batch object;
            batch_idx  - index of the abovementioned batch, int.
            
        Output:
        
            loss       - loss value for the particular mini-batch with images, tensor.
            
        """
        
        # Get images and their corresponding labels
        x, y = batch
        # Get logits
        logits = self(x)
        # Compute loss        
        loss = F.cross_entropy(logits, y)
        # Get indices of the logits with max value
        preds = torch.argmax(logits, dim = 1)
        # Compute accuracy score
        acc = self.accuracy(preds, y)
        pr = self.pr(preds, y)
        f1 = self.f1(preds, y)
        
        # Logs
        self.log("train_loss", loss, on_step = False, on_epoch = True, logger = True); self.log("train_acc", acc, on_step = False, on_epoch = True, logger = True);
        self.log("train_f1", f1, on_step = False, on_epoch = True, logger = True, sync_dist = True); self.log("train_pr", pr, on_step = False, on_epoch = True, logger = True, sync_dist = True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        """
        
        This function gets several parameters and conducts training step for a single batch.
        
        Parameters:
        
            batch      - a single batch of the dataloader, batch object;
            batch_idx  - index of the abovementioned batch, int.
            
        Output:
        
            loss       - loss value for the particular mini-batch with images, tensor.
            
        """
        
        # Get images and their corresponding labels
        x, y = batch
        # Get logits
        logits = self(x)
        # Compute loss
        loss = F.cross_entropy(logits, y)
        # Get indices of the logits with max value
        preds = torch.argmax(logits, dim = 1)
        # Compute accuracy score
        acc = self.accuracy(preds, y)
        pr = self.pr(preds, y)
        f1 = self.f1(preds, y)
        # Log
        self.log("valid_loss", loss, prog_bar = True); self.log("valid_acc", acc, prog_bar = True);
        self.log("valid_f1", f1, on_step = False, on_epoch = True, logger = True, sync_dist = True); self.log("valid_pr", pr, on_step = False, on_epoch = True, logger = True, sync_dist = True)
        
        return loss
    
    # Set the time when validation process is started
    def on_validation_epoch_start(self): self.validation_start_time = time()
    
    # Compute the time when validation process is finished
    def on_validation_epoch_end(self): self.validation_elapsed_time = time() - self.validation_start_time; self.validation_times.append(self.validation_elapsed_time); self.log("valid_time", self.validation_elapsed_time, prog_bar = True)
    
    # Get stats of the train and validation times
    def get_stats(self): return self.train_times, self.validation_times
    
class ImagePredictionLogger(Callback):

    """
    
    This class gets several parameters and visualizes several input images and predictions in the end of validation process.

    Parameters:

        val_samples       - validation samples, torch dataloader object;
        cls_names         - class names, list;
        num_samples       - number of samples to be visualized, int.
      
    """
    
    def __init__(self, val_samples, cls_names = None, num_samples = 4):
        super().__init__()
        # Get class arguments
        self.num_samples, self.cls_names = num_samples, cls_names
        # Extract images and their corresponding labels
        self.val_imgs, self.val_labels = val_samples
        
    def on_validation_epoch_end(self, trainer, pl_module):

        """
        
        This function gets several parameters and visualizes images with their predictions.

        Parameters:

            trainer      - trainer, pytorch lightning trainer object;
            pl_module    - model class, pytorch lightning module object.
        
        """
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        if self.cls_names != None:
            trainer.logger.experiment.log({
                "Sample Validation Prediction Results":[wandb.Image(x, caption=f"Predicted class: {self.cls_names[pred]}, Ground truth class: {self.cls_names[y]}") 
                               for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                     preds[:self.num_samples], 
                                                     val_labels[:self.num_samples])]
                })
