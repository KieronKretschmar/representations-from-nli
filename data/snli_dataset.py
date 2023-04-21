from torch.utils.data.dataset import Dataset
import tqdm
import nltk
import torch

class SNLIDataset(Dataset):
    def __init__(self, unprocessed_data, vocab) -> None:
        """A Dataset wrapper around a split of the SNLI Dataset.
        The words in hypotheses and premises are replaced with their embeddings once upon initialization.

        Args:
            vocab (Vocabulary): Vocabulary object holding word embeddings.
        """
        super().__init__()

        self.data = [(
            torch.stack([vocab.get_embedding(token) for token in nltk.word_tokenize(example["premise"])], dim=0),
            torch.stack([vocab.get_embedding(token) for token in nltk.word_tokenize(example["hypothesis"])], dim=0),
            example["label"])
            for example in tqdm.tqdm(unprocessed_data) if example["label"] > 0]      # Remove samples with broken labels < 0

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
