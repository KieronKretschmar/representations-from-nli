import os
import torch
from torch.utils.data import DataLoader
import pickle
from pathlib import Path
import numpy as np
import requests
from zipfile import ZipFile
from io import BytesIO
from datasets import load_dataset
import nltk
import pytorch_lightning as pl
import tqdm

from .snli_dataset import SNLIDataset
from .vocabulary import Vocabulary


class SNLIDataModule(pl.LightningDataModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.batch_size = args.batch_size
        self.rebuild_cache = args.rebuild_cache
        self.use_subset = args.use_subset
        self.unk_src = args.unk_src
        self.num_workers = args.num_workers
        self.cache_path = Path(args.cache_path)
        self.vocab_cache_path = self.cache_path / "vocab.pkl"
    
    def prepare_data(self) -> None:
        """Download the SNLI dataset and creates and caches Vocabulary and token2vec dictionary.
        """
        # Download prerequisites
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')


        # Try to load self.vocab and self.word_embeddor from cache 
        cached_vocab_exist = Path.exists(self.vocab_cache_path)
        # Create self.vocab and self.word_embeddor if necessary
        if self.rebuild_cache or not cached_vocab_exist:
            vocab = Vocabulary(use_subset = self.use_subset, unk_src=self.unk_src)
            vocab.build_from_snli()

            with open(self.vocab_cache_path, 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def setup(self, stage: str) -> None:
        self.snli_ds = load_dataset('snli')
        
        if self.use_subset:
            print("Using reduced snli dataset")
            for split in self.snli_ds:
                self.snli_ds[split] = self.snli_ds["test"]

        print("Loading Vocab from cache.")
        with open(self.vocab_cache_path, 'rb') as handle:
            vocab = pickle.load(handle)

        # Make sure the vocabulary uses the desired settings
        assert vocab.unk_src == self.unk_src, f"The cached vocabulary at {self.vocab_cache_path} uses unk_src {vocab.unk_src}, " \
            f"but {self.unk_src} was specified. Consider rebuilding cache with --rebuild-cache option."

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            print("Preparing training dataset...")
            self.train_ds = SNLIDataset(self.snli_ds["train"], vocab) 
            print("Preparing validation dataset...")
            self.val_ds = SNLIDataset(self.snli_ds["validation"], vocab) 

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_ds = SNLIDataset(self.snli_ds["test"], vocab) 

        # if stage == "predict":
        #     self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def collate_snli(self, batch):
        prems = [x[0] for x in batch]
        hypos = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        # store lengths required for packing later
        prem_lengths = [len(seq) for seq in prems]
        hypo_lengths = [len(seq) for seq in hypos]

        # pad prems and hypos
        prems_padded = torch.nn.utils.rnn.pad_sequence(prems, batch_first=True)
        hypos_padded = torch.nn.utils.rnn.pad_sequence(hypos, batch_first=True)

        return (prems_padded, prem_lengths), (hypos_padded, hypo_lengths), torch.tensor(labels)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_snli)

    def val_dataloader(self):
        val_dl = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_snli)
        return val_dl

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_snli)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=32)