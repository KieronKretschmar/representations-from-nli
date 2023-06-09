from .sentence_encoders import model_dict
import torch
import nltk

class Batcher():
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab

    def __call__(self, params, batch):
        """ Transforms a batch of text sentences into sentence embeddings.

        Args:
            params (_type_): _description_
            batch (list of str): List of sentences.
        """
       
        # Transform each sentence into a single string
        # Empty sentences will be replaced by the unknown token
        seqs = [" ".join([tokens for tokens in seq]) if len(seq) > 0 else self.vocab.unk_token for seq in batch]

        # Transform each sentence from string to a variable length tensor of word embeddings
        seqs = [torch.stack([self.vocab.get_embedding(token) for token in nltk.word_tokenize(seq)], dim=0) for seq in seqs]

        # Store sequence lengths for encoder
        seq_lengths = [len(seq) for seq in seqs]

        # Transform into a single padded tensor 
        seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        
        # Transform batch tensor into a single sentence embedding
        seq_embs = self.model(seqs, seq_lengths)   # (B, SeqEmb)

        seq_embs = seq_embs.detach().numpy()

        return seq_embs
