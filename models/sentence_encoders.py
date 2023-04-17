import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class BaselineModel(nn.Module):
    """Creates sentence embeddings as means of word embeddings.
    """

    def __init__(self, input_size = 300) -> None:
        super().__init__()
        self.output_size = input_size

    def forward(self, seqs, seq_lengths):
        """Returns mean along token axis.

        Args:
            seqs (tensor): A padded batch of sequences of shape (B, MaxSeqLen, WordEmb).
            seq_lengths (list of int): Lengths of the sequences before padding.

        Returns:
            tensor: Tensor of shape (B, WordEmb).
        """
        sums = torch.sum(seqs, dim=1)                                           # (B, WordEmb)
        # Device sums by variable lengths
        means = sums / torch.tensor(seq_lengths, device=sums.device)[:, None]   # (B, WordEmb)
        return means
    
class UnidirectionalLSTM(nn.Module):
    def __init__(self, input_size = 300, output_size = 2048) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = output_size
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            batch_first = True            
        )

    def forward(self, seqs, seq_lengths):
        """Returns final hidden state of the LSTM.

        Args:
            seqs (tensor): A padded batch of sequences of shape (B, MaxSeqLen, WordEmb).
            seq_lengths (list of int): Lengths of the sequences before padding.

        Returns:
            tensor: Final hidden state of shape (B, hidden_size)
        """
        x = pack_padded_sequence(seqs, seq_lengths, batch_first = True, enforce_sorted=False)

        output, (h_n, c_n) = self.lstm(x) #h_n has shape (1, B, hidden_size)
        out = h_n[0, ...] # (B, hidden_size)
        return out

        
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size = 300, output_size = 4096) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = int(output_size / 2)
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            bidirectional = True,
            batch_first = True
        )

    def forward(self, seqs, seq_lengths):
        """Returns concatenation of final forward and reverse hidden states of the LSTM.

        Args:
            seqs (tensor): A padded batch of sequences of shape (B, MaxSeqLen, WordEmb).
            seq_lengths (list of int): Lengths of the sequences before padding.

        Returns:
            tensor: Final hidden state of shape (B, 2*hidden_size)
        """
        x = pack_padded_sequence(seqs, seq_lengths, batch_first = True, enforce_sorted=False)

        output, (h_n, c_n) = self.lstm(x) #h_n has shape (1, B, 2*hidden_size)
        out = torch.cat((h_n[0], h_n[1]), dim=1) # (B, 2*hidden_size)
        return out
    
        
class BidirectionalMaxPoolLSTM(nn.Module):
    def __init__(self, input_size = 300, output_size = 4096) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = output_size / 2
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            bidirectional = True,
            batch_first = True
        )

    def forward(self, seqs, seq_lengths):
        """Returns sentence representation obtained by max pooling over features hidden states of the LSTM.

        Args:
            seqs (tensor): A padded batch of sequences of shape (B, MaxSeqLen, WordEmb).
            seq_lengths (list of int): Lengths of the sequences before padding.

        Returns:
            tensor: Final hidden state of shape (B, 2*hidden_size)
        """
        seqs = pack_padded_sequence(seqs, seq_lengths, batch_first=True, enforce_sorted=False)

        output, (h_n, c_n) = self.lstm(seqs)
        output_padded = pad_packed_sequence(output, batch_first=True) # (B, MaxSeqLen, 2 * hidden_size)

        # Ignore paddings when taking max by overwriting them with large negative value
        with torch.no_grad():
            output_padded[output_padded==0] = -1e8
            
        out = torch.max(output_padded, dim = 1) # (B, 2 * hidden_size)
        return out

model_dict = {
    "baseline" : BaselineModel,
    "unilstm": UnidirectionalLSTM,
    "bilstm": BidirectionalLSTM,
    "bimaxlstm": BidirectionalMaxPoolLSTM
    }

def create_encoder(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f"Unknown model name \"{model_name}\". Available models are: {str(model_dict.keys())}"