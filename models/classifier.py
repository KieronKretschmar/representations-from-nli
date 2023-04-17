import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import pytorch_lightning as pl


class Classifier(nn.Module):
    """The 3-layer classifier network as proposed by Bowman et al. in arxiv.org/abs/1705.02364v5 Figure 3.
    """
    def __init__(self, input_dim, hidden_dim = 512, n_classes = 3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.n_classes)
        )

    def forward(self, x):
        """Returns logits.

        Args:
            x (tensor): A batch of sentence representations with dimension (Batch_dim, input_dim)

        Returns:
            tensor: Tensor holding logits with dimension (Batch_dim, n_classes)
        """
        return self.net(x)