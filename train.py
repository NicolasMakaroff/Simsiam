from attr import evolve
from model_params import SimSIAMParams
from simsiam import SelfSupervisedMethod

import os
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb

os.environ["DATA_PATH"] = "../data"

def train_model():
    params = SimSIAMParams()
    model = SelfSupervisedMethod(params)
    wandb_logger = WandbLogger(project="vicreg")
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=200, check_val_every_n_epoch=1, num_sanity_val_steps=0 ,logger = wandb_logger)
    trainer.fit(model)
    trainer.save_checkpoint("example.ckpt")

if __name__ == "__main__":
    sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'maximize', 
            'name': 'validation_accuracy'
            },
        'parameters': {
            'lr': {'max': 0.001, 'min': 0.0009},
            'weight_decay': {'max': 1e-3, 'min': 1e-5},
        }
    }
    #sweep_id = wandb.sweep(sweep=sweep_configuration, project="vicreg")
    #wandb.agent(sweep_id=sweep_id, function=train_model, count=20)
    train_model()