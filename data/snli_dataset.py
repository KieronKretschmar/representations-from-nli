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

        # self.data = []
        # for index in tqdm.tqdm(range(len(unprocessed_data["premise"]))):
        #     premise = word_embeddor.embed(unprocessed_data["premise"][index])
        #     hypothesis = word_embeddor.embed(unprocessed_data["hypothesis"][index])

            
        #     premise = word_embeddor.embed(unprocessed_data["premise"][index])
        #     hypothesis = word_embeddor.embed(unprocessed_data["hypothesis"][index])
            # label = unprocessed_data["label"][index]
            # if label < 0:
            #     print(f"Invalid label found at index {index}. Dismissing sample.")
            #     continue
            # self.data.append((premise, hypothesis, label))

        # self.data = []
        # for i in tqdm.tqdm(range(0, len(unprocessed_data["premise"]), self.chunk_size)):
        #     premise = word_embeddor.embed(unprocessed_data["premise"][i:i + self.chunk_size])
        #     hypothesis = word_embeddor.embed(unprocessed_data["hypothesis"][i:i + self.chunk_size])
        #     label = torch.tensor(unprocessed_data["label"][i:i + self.chunk_size]).to(word_embeddor.device)

        #     valid_mask = label > 0
        #     if not valid_mask.all():
        #         print(f"Invalid label found at chunk index {i}. Dismissing sample.")
        #         continue

        #     self.data.append((premise, hypothesis, label))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
