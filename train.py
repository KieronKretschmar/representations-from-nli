import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import requests
from zipfile import ZipFile
from io import BytesIO
from datasets import load_dataset
import nltk
import pytorch_lightning as pl
import pickle
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar, EarlyStopping
import argparse
from datetime import datetime

from models import NLIModel
from data import SNLIDataModule

LOG_PATH = Path("./logs")
LOG_PATH.mkdir(exist_ok=True)
CHECKPOINT_PATH = Path("./checkpoints")
CHECKPOINT_PATH.mkdir(exist_ok=True)
BEST_ENCODER_CHECKPOINT_PATH = Path("./checkpoints/best")
BEST_ENCODER_CHECKPOINT_PATH.mkdir(exist_ok=True)
CACHE_PATH = Path("./cache")
CACHE_PATH.mkdir(exist_ok=True)


def train_model(datamodule, encoder_name, save_name=None, use_wandb = False, **model_kwargs):
    """
    Method adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html

    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl.seed_everything(42)


    encoder_names = ["baseline", "unilstm", "bilstm", "bimaxlstm"] if encoder_name == 'all' else [encoder_name]
    for encoder_name in encoder_names:
            
        if save_name is None:
            save_name = encoder_name

        try:
            logger = pl.loggers.WandbLogger(project="representations-from-nli", name=save_name + "_" + datetime.now().strftime("%d/%m/%Y-%H:%M:%S"), save_dir=LOG_PATH)

        except:
            print("Logging offline!")
            logger = pl.loggers.WandbLogger(project="representations-from-nli", name=save_name + "_" + datetime.now().strftime("%d/%m/%Y-%H:%M:%S"), save_dir=LOG_PATH, offline=True)

        # Create a PyTorch Lightning trainer with the generation callback
        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),                         # Where to save models
                            accelerator="gpu" if str(device).startswith("cuda") else "cpu",                     # We run on a GPU (if possible)
                            devices=1,                                                                          # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
                            max_epochs=25,                                                                      # How many epochs to train for if no patience is set
                            callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                        LearningRateMonitor("epoch"),                                           # Log learning rate every epoch
                                        TQDMProgressBar(refresh_rate=100),                                      # Don't flood SLURM
                                        EarlyStopping(monitor="lr", mode="min", stopping_threshold=1e-5),]      # Stop when lr goes below 1e-5
                            logger=logger,                                                                      # Pass wandb logger
                            enable_progress_bar=True)                                                           # Set to False if you do not want a progress bar

        pl.seed_everything(42) # To be reproducable
        model = NLIModel(encoder_name=encoder_name, **model_kwargs)
        trainer.fit(model, datamodule=datamodule)
        print(f"Done fitting {encoder_name}.")
        best_model = NLIModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

        # Store best model for later evaluation
        best_encoder_path = BEST_ENCODER_CHECKPOINT_PATH / encoder_name
        print(f"Saving encoder to {best_encoder_path}...")
        torch.save(best_model.encoder, best_encoder_path)

def main():
    

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder", 
        choices=["all", "baseline", "unilstm", "bilstm", "bimaxlstm"], 
        default="baseline",
        required=False
    )
    parser.add_argument(
        "--save-name", 
        default=None,
        required=False
    )
    parser.add_argument("--rebuild-cache", help="Rebuild vocabulary even if cached version is available.",
                    action="store_true",
                    default=False)
    
    parser.add_argument("--use-subset", help="Use subset of the data for training.",
                    action="store_true",
                    default=False)

    args = parser.parse_args()

    if False:
        print("Debugging...")
        args = argparse.Namespace(
            encoder="unilstm",
            save_name=None,
            rebuild_cache=True,
            use_subset=True
        )

    print("args: ", args)


    datamodule = SNLIDataModule(rebuild_cache=args.rebuild_cache, use_subset=args.use_subset, cache_path=CACHE_PATH)

    train_model(
        datamodule,
        args.encoder,
        save_name=args.save_name)


if __name__ == '__main__':
    main()
