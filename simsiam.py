import copy
import math
from functools import partial
from typing import List, Optional, Union

import attr
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
import torch.nn.functional as F
import torchvision.models as models
from pytorch_lightning.utilities.parsing import AttributeDict
from torch.utils.data import DataLoader

import utils
from sklearn.linear_model import LogisticRegression
from utils import SimSiam, M_O
from model_params import ModelParams
from lars import LARS 
import numpy as np

import wandb

class SelfSupervisedMethod(pl.LightningModule):
    model: torch.nn.Module
    dataset: utils.DatasetBase
    hparams: AttributeDict
    embeding_dim: Optional[int] = None

    def __init__(
            self,
            hparams: Union[ModelParams, dict, None] = None,
            **kwargs,
    ):
        super().__init__()

        if hparams is None:
            hparams = self.params(**kwargs)
        elif isinstance(hparams, dict):
            hparams = self.params(**hparams, **kwargs)

        if isinstance(self.hparams, AttributeDict):
            self.hparams.update(AttributeDict(attr.asdict(hparams)))
        else:
            self.hparams = AttributeDict(attr.asdict(hparams))
        
        # Create encoder model
        self.model = SimSiam(base_encoder=models.__dict__['resnet18'], dim=2048, pred_dim=512) #utils.get_encoder(hparams.encoder_arch)

        # Create dataset
        self.dataset = utils.get_dataset(hparams)

        self.criterion = torch.nn.CosineSimilarity(dim=1)

        #self.decoder_model = utils.get_decoder(hparams.decoder_arch)

        self.sklearn_classifier = LogisticRegression(max_iter=100, solver='liblinear')

    def _get_embeddings(self, x):
        
        bsz, nd, nc, nh, nw = x.shape
        assert nd == 2, "Input must be a pair of images"
        im_q = x[:, 0].contiguous()
        im_k = x[:, 1].contiguous()

        # Get the embedding
        p1, p2, z1, z2, h1, h2, x1_hat, x2_hat = self.model(im_q, im_k)#, mean_q, logvar_q, mean_k, logvar_k = self.model(im_q, im_k)
        #z_k = self.model(im_k)

        # Project the embedding
        #p_q = self.projection_model(z_q)
        #p_k = self.projection_model(z_k)

        # Predict the embedding
        #z_q_hat = self.prediction_model(p_q)
        #z_k_hat = self.prediction_model(p_k)

        # Decode the embedding
        #x_q_hat = self.decoder_model(z_q)
        #x_k_hat = self.decoder_model(z_k)

        return p1, p2, z1, z2, h1, h2, x1_hat, x2_hat#, mean_q, logvar_q, mean_k, logvar_k
    
    def _get_simsiam_loss(self, p1, p2, z1, z2):
        # Compute the loss
        loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5 + 1
        return { 
            "loss": loss,
        }
    
    def _get_reconstruction_loss(self, x1, x2, x1_hat, x2_hat):
        # Compute the loss
        loss = 0.5 * (F.mse_loss(x1_hat, x1) + F.mse_loss(x2_hat, x2))
        #0.5 * (F.cross_entropy(x1_hat, x1) + F.cross_entropy(x2_hat, x2)) #(F.mse_loss(x1_hat, x1) + F.mse_loss(x2_hat, x2)) * 0.5
        return {
            "loss": loss,
        }
    
    def _get_kl_loss(self, mean_q, logvar_q, mean_k, logvar_k):
        # Compute the loss
        loss =  0.5 * torch.mean(logvar_q**2 + mean_q**2 - torch.log(logvar_q) - 1/2) + 0.5 * torch.mean(logvar_k**2 + mean_k**2 - torch.log(logvar_k) - 1/2)
        return {
            "loss": 0.5 * loss,
        }
    
    def forward(self, x1, x2):
        return self.model(x1, x2)
    
    def training_step(self, batch, batch_idx):
        x, class_labels = batch 
        x1 = x[:, 0].contiguous().detach()
        x2 = x[:, 1].contiguous().detach()
        p1, p2, z1, z2, _, _, x1_hat, x2_hat = self._get_embeddings(x) #, mean_1, logvar_1, mean_2, logvar_2 = self._get_embeddings(x)

        # Get the loss
        loss_dict = self._get_simsiam_loss(p1, p2, z1, z2)
        reconctruction_loss_dict = self._get_reconstruction_loss(x1, x2, x1_hat, x2_hat)
        #kl_loss_dict = self._get_kl_loss(mean_1, logvar_1, mean_2, logvar_2)
        w = 0.2
        beta = 0.0
        loss = (1 - w) * loss_dict["loss"] + w * (reconctruction_loss_dict["loss"])# + beta * kl_loss_dict["loss"])
        self.log("sim_loss", loss_dict["loss"].item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("rec_loss", reconctruction_loss_dict["loss"].item(), on_step=True, on_epoch=True, prog_bar=True)
        #self.log("kl_loss", kl_loss_dict["loss"].item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(loss_dict)
        #pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        #print(pytorch_total_params)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        x, class_labels = batch
        with torch.no_grad():
            _, _, _, _, emb, _, _, _ = self.model(x, x) #, _, _, _, _  = self.model(x, x)
        return {"emb": emb, "labels": class_labels}
    
    def validation_epoch_end(self, outputs):
        embeddings = torch.cat([x['emb'] for x in outputs]).cpu().detach().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).cpu().detach().numpy()
        #np.save("embeddings.npy", embeddings)
        num_split_linear = embeddings.shape[0] // 2
        self.sklearn_classifier.fit(embeddings[:num_split_linear], labels[:num_split_linear])
        train_acc = self.sklearn_classifier.score(embeddings[:num_split_linear], labels[:num_split_linear])*100
        valid_accuracy = self.sklearn_classifier.score(embeddings[num_split_linear:], labels[num_split_linear:])*100

        log_data = {
            "epoch": self.current_epoch,
            "train_acc": train_acc,
            "valid_acc": valid_accuracy,
        }
        wandb.log({'validation_accuracy': valid_accuracy})
        m_r, m_o, r, o = M_O(embeddings)
        print(f"Epoch {self.current_epoch}: Train acc: {train_acc:.2f}%, Valid acc: {valid_accuracy:.2f}%, M_R: {m_r.mean()*100:.2f}%, M_O: {m_o.mean()*100:.2f}%")
        self.log_dict(log_data)
        #wandb.log({
        #'val_loss': valid_accuracy
        #})

    def configure_optimizers(self):
        regular_parameter = []
        regular_paramters_name = []
        excluded_parameter = []
        excluded_paramters_name = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if "model" in name:
                    regular_parameter.append(param)
                    regular_paramters_name.append(name)
                else:
                    excluded_parameter.append(param)
                    excluded_paramters_name.append(name)
        param_groups = [
            {"params": regular_parameter, "names": regular_paramters_name, "use_lars": True},
            {
                "params": excluded_parameter,
                "names": excluded_paramters_name,
                "weight_decay": 0.0,
                "use_lars": False,
            }
        ]
        if self.hparams.optimizer_name == "lars":
            optimizer = partial(LARS, warmup_epochs=self.hparams.lars_warmup_epochs, eta=self.hparams.lars_eta)
        elif self.hparams.optimizer_name == "sgd":
            optimizer = torch.optim.SGD
        elif self.hparams.optimizer_name == "adam":
            optimizer = torch.optim.Adam
        else:
            raise NotImplementedError(f"{self.hparams.optimizer_name} is not implemented")
        if self.hparams.optimizer_name == "adam":
            encodings_optimizer = optimizer(param_groups, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            encodings_optimizer = optimizer(param_groups, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(encodings_optimizer, T_max=self.hparams.max_epochs, eta_min=0)#self.hparams.final_lr_schedule_value)
        return [encodings_optimizer], [self.lr_scheduler]
    
    def prepare_data(self) -> None:
        self.dataset.get_train()
        self.dataset.get_validation()

    def train_dataloader(self):
        return DataLoader(
            self.dataset.get_train(),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            pin_memory=self.hparams.pin_data_memory,
            drop_last=self.hparams.drop_last_batch,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.dataset.get_validation(),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            pin_memory=self.hparams.pin_data_memory,
            drop_last=self.hparams.drop_last_batch,
        )
    
    @classmethod
    def params(cls, **kwargs) -> ModelParams:
        return ModelParams(**kwargs)