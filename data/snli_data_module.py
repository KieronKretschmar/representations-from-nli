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
    def __init__(self, vocab_src = "snli", rebuild_cache = False, num_workers = 3, batch_size=64, use_subset = False) -> None:
        super().__init__()
        self.vocab_src = vocab_src
        self.batch_size = batch_size
        self.rebuild_cache = rebuild_cache
        self.num_workers = num_workers
        self.vocab_cache_path = Path("cache/vocab.pickle")
        self.token2vec_cache_path = Path("cache/token2vec.pickle")
        self.use_subset = use_subset
    
    def prepare_data(self) -> None:
        """Download the SNLI dataset and creates and caches Vocabulary and token2vec dictionary.
        """
        # Download prerequisites
        snli_ds = load_dataset('snli')

        if self.use_subset:
            print("Using reduced snli dataset")
            for split in snli_ds:
                snli_ds[split] = snli_ds["test"]

        # Try to load self.vocab and self.word_embeddor from cache 
        cached_vocab_embeddor_exist = Path.exists(self.token2vec_cache_path) and Path.exists(Path(self.vocab_cache_path))
        # Create self.vocab and self.word_embeddor if necessary
        if self.rebuild_cache or not cached_vocab_embeddor_exist:
            print("Building data objects.")            
            # Get glove txt file
            glove_txt_path = Path(r"data/glove/glove.840B.300d.txt")
            if not Path.exists(glove_txt_path):
                print("Downloading GloVe...")            
                # Download and unzip GloVe
                url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
                response = requests.get(url)
                with ZipFile(BytesIO(response.content)) as file:
                    file.extractall(glove_txt_path.parent)
                        
            # Read GloVe from txt file and store all vectors in dictionary
            token2vec = {}
            with open(glove_txt_path, 'r', encoding="utf-8") as file:
                print("Reading GloVe embeddings...")
                for i, line in tqdm.tqdm(enumerate(file)):
                    if self.use_subset and i >= 100:
                        print("Using reduced glove dataset")
                        break
                    vals = line.strip().split(' ')
                    word = vals[0]
                    vec = torch.tensor(np.array(vals[1:], dtype="float32"))

                    token2vec[word] = vec

            # Determine desired vocab tokens specified by vocab_src 
            desired_vocab_tokens = []
            if self.vocab_src == "snli":
                for split in snli_ds:
                    print(f"Loading split {split} for building vocab...")
                    for sentence in tqdm.tqdm(snli_ds[split]["premise"]):
                        desired_vocab_tokens += nltk.word_tokenize(sentence.lower())
                    for sentence in tqdm.tqdm(snli_ds[split]["hypothesis"]):
                        desired_vocab_tokens += nltk.word_tokenize(sentence.lower())

            # Create vocabulary with all words that are in intersection of GloVe and desired vocab tokens
            vocab = Vocabulary()
            print("Building vocab...")
            for token in tqdm.tqdm(list(set(desired_vocab_tokens))):
                if token in token2vec:
                    vocab.add_word(token)

            # for unknown tokens, we use the average word vector of GloVe 840B 300d
            # taken from https://stackoverflow.com/questions/49239941/what-is-unk-in-the-pretrained-glove-vector-files-e-g-glove-6b-50d-txt
            average_glove_string = '0.22418134 -0.28881392 0.13854356 0.00365387 -0.12870757 0.10243822 0.061626635 0.07318011 -0.061350107 -1.3477012 0.42037755 -0.063593924 -0.09683349 0.18086134 0.23704372 0.014126852 0.170096 -1.1491593 0.31497982 0.06622181 0.024687296 0.076693475 0.13851812 0.021302193 -0.06640582 -0.010336159 0.13523154 -0.042144544 -0.11938788 0.006948221 0.13333307 -0.18276379 0.052385733 0.008943111 -0.23957317 0.08500333 -0.006894406 0.0015864656 0.063391194 0.19177166 -0.13113557 -0.11295479 -0.14276934 0.03413971 -0.034278486 -0.051366422 0.18891625 -0.16673574 -0.057783455 0.036823478 0.08078679 0.022949161 0.033298038 0.011784158 0.05643189 -0.042776518 0.011959623 0.011552498 -0.0007971594 0.11300405 -0.031369694 -0.0061559738 -0.009043574 -0.415336 -0.18870236 0.13708843 0.005911723 -0.113035575 -0.030096142 -0.23908928 -0.05354085 -0.044904727 -0.20228513 0.0065645403 -0.09578946 -0.07391877 -0.06487607 0.111740574 -0.048649278 -0.16565254 -0.052037314 -0.078968436 0.13684988 0.0757494 -0.006275573 0.28693774 0.52017444 -0.0877165 -0.33010918 -0.1359622 0.114895485 -0.09744406 0.06269521 0.12118575 -0.08026362 0.35256687 -0.060017522 -0.04889904 -0.06828978 0.088740796 0.003964443 -0.0766291 0.1263925 0.07809314 -0.023164088 -0.5680669 -0.037892066 -0.1350967 -0.11351585 -0.111434504 -0.0905027 0.25174105 -0.14841858 0.034635577 -0.07334565 0.06320108 -0.038343467 -0.05413284 0.042197507 -0.090380974 -0.070528865 -0.009174437 0.009069661 0.1405178 0.02958134 -0.036431845 -0.08625681 0.042951006 0.08230793 0.0903314 -0.12279937 -0.013899368 0.048119213 0.08678239 -0.14450377 -0.04424887 0.018319942 0.015026873 -0.100526 0.06021201 0.74059093 -0.0016333034 -0.24960588 -0.023739101 0.016396184 0.11928964 0.13950661 -0.031624354 -0.01645025 0.14079992 -0.0002824564 -0.08052984 -0.0021310581 -0.025350995 0.086938225 0.14308536 0.17146006 -0.13943303 0.048792403 0.09274929 -0.053167373 0.031103406 0.012354865 0.21057427 0.32618305 0.18015954 -0.15881181 0.15322933 -0.22558987 -0.04200665 0.0084689725 0.038156632 0.15188617 0.13274793 0.113756925 -0.095273495 -0.049490947 -0.10265804 -0.27064866 -0.034567792 -0.018810693 -0.0010360252 0.10340131 0.13883452 0.21131058 -0.01981019 0.1833468 -0.10751636 -0.03128868 0.02518242 0.23232952 0.042052146 0.11731903 -0.15506615 0.0063580726 -0.15429358 0.1511722 0.12745973 0.2576985 -0.25486213 -0.0709463 0.17983761 0.054027 -0.09884228 -0.24595179 -0.093028545 -0.028203879 0.094398156 0.09233813 0.029291354 0.13110267 0.15682974 -0.016919162 0.23927948 -0.1343307 -0.22422817 0.14634751 -0.064993896 0.4703685 -0.027190214 0.06224946 -0.091360025 0.21490277 -0.19562101 -0.10032754 -0.09056772 -0.06203493 -0.18876675 -0.10963594 -0.27734384 0.12616494 -0.02217992 -0.16058226 -0.080475815 0.026953284 0.110732645 0.014894041 0.09416802 0.14299914 -0.1594008 -0.066080004 -0.007995227 -0.11668856 -0.13081996 -0.09237365 0.14741232 0.09180138 0.081735 0.3211204 -0.0036552632 -0.047030564 -0.02311798 0.048961394 0.08669574 -0.06766279 -0.50028914 -0.048515294 0.14144728 -0.032994404 -0.11954345 -0.14929578 -0.2388355 -0.019883996 -0.15917352 -0.052084364 0.2801028 -0.0029121689 -0.054581646 -0.47385484 0.17112483 -0.12066923 -0.042173345 0.1395337 0.26115036 0.012869649 0.009291686 -0.0026459037 -0.075331464 0.017840583 -0.26869613 -0.21820338 -0.17084768 -0.1022808 -0.055290595 0.13513643 0.12362477 -0.10980586 0.13980341 -0.20233242 0.08813751 0.3849736 -0.10653763 -0.06199595 0.028849555 0.03230154 0.023856193 0.069950655 0.19310954 -0.077677034 -0.144811'
            unk_vector = torch.tensor(np.array(average_glove_string.split(" "), dtype="float32"))
            token2vec[vocab.unk_token] = unk_vector

            # Cache for future runs
            with open(self.token2vec_cache_path, 'wb') as handle:
                pickle.dump(token2vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(self.vocab_cache_path, 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def setup(self, stage: str) -> None:
        self.snli_ds = load_dataset('snli')
        
        if self.use_subset:
            print("Using reduced snli dataset")
            for split in self.snli_ds:
                self.snli_ds[split] = self.snli_ds["test"]

        print("Loading token2vec from cache.")
        with open(self.token2vec_cache_path, 'rb') as handle:
            token2vec = pickle.load(handle)

        print("Loading Vocab from cache.")
        with open(self.vocab_cache_path, 'rb') as handle:
            vocab = pickle.load(handle)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            print("Preparing training dataset...")
            self.train_ds = SNLIDataset(self.snli_ds["train"], token2vec, token2vec[vocab.unk_token]) 
            print("Preparing validation dataset...")
            self.val_ds = SNLIDataset(self.snli_ds["validation"], token2vec, token2vec[vocab.unk_token]) 

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_ds = SNLIDataset(self.snli_ds["test"], token2vec, token2vec[vocab.unk_token]) 

        # if stage == "predict":
        #     self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def collate_snli(self, batch):
        print("Collating...\n")
        prems = [x[0] for x in batch]
        hypos = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        # store lengths required for packing later
        prem_lengths = [len(seq) for seq in prems]
        hypo_lengths = [len(seq) for seq in hypos]

        # pad prems and hypos
        prems_padded = torch.nn.utils.rnn.pad_sequence(prems, batch_first=True)
        hypos_padded = torch.nn.utils.rnn.pad_sequence(hypos, batch_first=True)

        print("Collating finished! \n")
        return (prems_padded, prem_lengths), (hypos_padded, hypo_lengths), torch.tensor(labels)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_snli)

    def val_dataloader(self):
        print("Creating val dataloader...")
        val_dl = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_snli)
        print("Creating val dataloader finished!")
        return val_dl

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_snli)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=32)