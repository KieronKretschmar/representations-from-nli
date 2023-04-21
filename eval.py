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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-checkpoint-path", 
        required=True
    )

    parser.add_argument(
        "--vocab-path", 
        required=True
    )
    
    parser.add_argument(
        "--eval-config",
        choices=["default", "prototyping"],
        default="prototyping"
    )


    args = parser.parse_args()

    if False:
        print("Debugging...")
        args = argparse.Namespace(
            encoder_checkpoint_path="./checkpoints/best/unilstm.ckpt",
            vocab_path="./cache/vocab.pickle",
            senteval_data_path="./data/SentEval",
            eval_config="prototyping"
        )


    print(f"Loading encoder from {args.encoder_checkpoint_path}")
    torch.from_file(args.encoder_checkpoint_path)
    encoder = torch.load(args.encoder_checkpoint_path)
    encoder.eval()
    encoder_fname = Path(args.encoder_checkpoint_path).stem

    print(f"Loading Vocab from {args.encoder_checkpoint_path}")
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
        
    # Choose the same tasks as in Conneau et al. 2018
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

    save_results_path = SENTEVAL_RESULTS_PATH / (encoder_fname + ".pkl")
    with open(save_results_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Evaluation complete. Results saved to {save_results_path}!")

    
    
if __name__ == '__main__':
    main()