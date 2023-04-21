import os
import sys
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

import senteval

from models import NLIModel, Batcher
from data import SNLIDataModule

SENTEVAL_MODULE_PATH = Path(os.getcwd()) / "SentEval"
SENTEVAL_DATA_PATH = SENTEVAL_MODULE_PATH / "data"
SENTEVAL_RESULTS_PATH = Path("./senteval_results")
SENTEVAL_RESULTS_PATH.mkdir(exist_ok=True)
CACHE_PATH = Path("./cache")
CACHE_PATH.mkdir(exist_ok=True)

def eval_snli(args):
    print(f"Loading encoder from {args.encoder_checkpoint_path}")
    torch.from_file(args.encoder_checkpoint_path)
    encoder = torch.load(args.encoder_checkpoint_path)
    encoder.eval()

    snli_datamodule = SNLIDataModule(args)

    trainer = pl.Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",                          # We run on a GPU (if possible)
                devices=1,                                                                          # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
                enable_progress_bar=True)                                                           # Set to False if you do not want a progress bar

    nli_model = NLIModel(encoder=encoder)

    pl.seed_everything(42) # To be reproducable
    trainer.test(nli_model, datamodule=snli_datamodule)

def eval_senteval(args):
    
    print(f"Loading encoder from {args.encoder_checkpoint_path}")
    encoder = torch.load(args.encoder_checkpoint_path)
    encoder.eval()

    print(f"Loading Vocab from {args.vocab_path}")
    with open(args.vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)

    # Build SentEval params based on configurations offered in the SentEval repository
    if args.eval_config == "default":
        params = {'task_path': str(SENTEVAL_DATA_PATH), 'usepytorch': torch.cuda.is_available(), 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                        'tenacity': 5, 'epoch_size': 4}
    
    elif args.eval_config == "prototyping":
        params = {'task_path': str(SENTEVAL_DATA_PATH), 'usepytorch': torch.cuda.is_available(), 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                        'tenacity': 3, 'epoch_size': 2}
        
    # Choose the same tasks as in Conneau et al. 2017
    task_ids = [
        'MR',
        'CR',
        'SUBJ',
        'MPQA',
        'SST2',
        'TREC',
        'MRPC',
        'SICKRelatedness',
        'SICKEntailment',
        'STS14']
    
    # Create batcher which wraps the model to fit the SentEval interface
    batcher = Batcher(encoder, vocab)

    # Create senteval
    se = senteval.SE(params, batcher)

    # Evaluate
    print(f"Starting evaluation on encoder from {args.encoder_checkpoint_path}")
    results = se.eval(task_ids)
    print("Finished evaluation!")
    print(results)

    return results

def main():
    parser = argparse.ArgumentParser()

    # General evaluation args
    parser.add_argument(
        "--encoder-checkpoint-path",
        required=True
    )

    parser.add_argument(
        "--eval-task",
        choices=["senteval", "snli"],
        default="senteval"
    )
    
    # SNLI evaluation args
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--rebuild-cache",
        help="Rebuild vocabulary even if cached version is available.",
        action="store_true",
        default=False)
    
    parser.add_argument(
        "--use-subset",
        help="Use subset of the data for training.",
        action="store_true",
        default=False)

    parser.add_argument(
        "--unk-src",
        choices=["average", "zero"],
        default="average",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--cache-path",
        default="./cache",
    )

    # Args for evaluation with SentEval
    parser.add_argument(
        "--results-save-name",
        required=True
    )

    parser.add_argument(
        "--eval-config",
        choices=["default", "prototyping"],
        default="prototyping"
    )

    parser.add_argument(
        "--vocab-path",
        default="./cache/vocab.pkl",
    )

    args = parser.parse_args()

    if args.eval_task == "senteval":
        senteval_results = eval_senteval(args)
        senteval_save_results_path = SENTEVAL_RESULTS_PATH / args.results_save_name
        if not senteval_save_results_path.suffix:
            senteval_save_results_path = senteval_save_results_path.with_suffix(".pkl")
        with open(senteval_save_results_path, 'wb') as handle:
            pickle.dump(senteval_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Evaluation complete! Results saved to {senteval_save_results_path}.")

    elif args.eval_task == "snli":
        snli_results = eval_snli(args)
        print(f"Evaluation complete! Please obtain the results from logs.")

if __name__ == '__main__':
    main()