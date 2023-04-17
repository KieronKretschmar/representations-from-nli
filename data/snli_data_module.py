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
    def __init__(self, vocab_src = "snli", rebuild_cache = False, num_workers = 3, batch_size=64, use_subset = False, cache_path = Path("./cache")) -> None:
        super().__init__()
        self.vocab_src = vocab_src
        self.batch_size = batch_size
        self.rebuild_cache = rebuild_cache
        self.num_workers = num_workers
        self.cache_path = cache_path
        self.vocab_cache_path = self.cache_path / "vocab.pickle"
        self.use_subset = use_subset
    
    def prepare_data(self) -> None:
        """Download the SNLI dataset and creates and caches Vocabulary and token2vec dictionary.
        """
        # Download prerequisites
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')


        # Try to load self.vocab and self.word_embeddor from cache 
        cached_vocab_exist = Path.exists(Path(self.vocab_cache_path))
        # Create self.vocab and self.word_embeddor if necessary
        if self.rebuild_cache or not cached_vocab_exist:
            vocab = Vocabulary()
            vocab.build_from_snli(self.use_subset, unk_src="average")

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