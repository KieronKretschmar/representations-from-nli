import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import pytorch_lightning as pl
import time
from .classifier import Classifier
from .sentence_encoders import create_encoder

class NLIModel(pl.LightningModule):
    """A model to learn sentence representations from natural language inference as described by Conneau et al. in arxiv.org/abs/1705.02364v5.
    The model consists of a sequence embedding module followed by a 3-way classification module. 
    """
    def __init__(self, encoder_name, encoder_params = {}):
        super().__init__()

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()

        self.encoder_name = encoder_name
        # Initialize sequence embedding module
        self.encoder = create_encoder(encoder_name, encoder_params)
        # Initialize classification module
        self.task_emb_size = self.encoder.output_size * 4
        self.classifier = Classifier(self.task_emb_size)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        # self.example_input_array = (torch.zeros((1, 3, 300), dtype=torch.float32), [3], torch.zeros((1, 3, 300), dtype=torch.float32), [3])

    def forward(self, batch):
        """Takes premises and hypothesis and returns logits for each class.

        Args:
            batch (tuple): A batch containing 4 elements: 
                premises (tensor): padded tensor with each premise's word embeddings
                premise_lengths (list of ints): lengths of the premises
                hypotheses (tensor): padded tensor with each hypothesis' word embeddings
            
        Returns:
            tensor: A batch of logits.
        """
        (prems, prem_lengths), (hypos, hypo_lengths), labels = batch

        # Get sentence representations
        prem_embs = self.encoder(prems, prem_lengths)   # (B, SeqEmb)
        hypo_embs = self.encoder(hypos, hypo_lengths)   # (B, SeqEmb)

        # Use both sentence representations to create task representations
        task_embs = torch.cat((prem_embs, hypo_embs, torch.abs(prem_embs - hypo_embs), prem_embs * hypo_embs), dim=1) # (B, 2 * SeqEmb + SeqEmb + SeqEmb) = (B, 4 * SeqEmb)

        # Classify task representations
        logits = self.classifier(task_embs) # (B, 3)

        return logits

    def configure_optimizers(self):
        """Optimization according to section 3.3 of arxiv.org/abs/1705.02364v5:

        "For all our models trained on SNLI, we use SGD
        with a learning rate of 0.1 and a weight decay of
        0.99. At each epoch, we divide the learning rate
        by 5 if the dev accuracy decreases."
        """
        optimizer = optim.SGD(self.parameters(), lr=0.1)
        
        step_sched = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.99)
        step_sched_config = {
            "scheduler": step_sched,
            "interval": "epoch",
            "frequency": 1
        }

        plateau_sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            # reduce LR if accuracy _decreases_
            mode="max", 
            # reduce immediately after an epoch without improvement
            patience=0,
            # reduce the learning rate by factor of 5
            factor=0.2
            )
        plateau_sched_config = {
            "scheduler": plateau_sched,
            "monitor": "val_acc",
            "interval": "epoch",
            "frequency": 1
        }

        return [optimizer], [step_sched_config, plateau_sched_config]
    
    # Shameless copy-paste-adapt from Philipp Lippe's tutorial at
    # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html
    def training_step(self, batch, batch_idx):
        # "batch" is the output of the data loader.
        (prems, prem_lengths), (hypos, hypo_lengths), labels = batch
        preds = self.forward(batch)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss  # Return tensor to call ".backward" on
    

    def validation_step(self, batch, batch_idx):
        (prems, prem_lengths), (hypos, hypo_lengths), labels = batch
        preds = self.forward(batch).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', acc.item(), on_epoch=True)

    def test_step(self, batch, batch_idx):
        (prems, prem_lengths), (hypos, hypo_lengths), labels = batch
        input_batch = (prems, prem_lengths, hypos, hypo_lengths)
        preds = self.forward(input_batch).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('test_acc', acc.item(), on_epoch=True)