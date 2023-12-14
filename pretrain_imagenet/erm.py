import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from encoder_cnn import IMG_ENCODE_SIZE, EncoderCNN
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchmetrics import Accuracy


N_CLASSES = 1000


class ERM(pl.LightningModule):
    def __init__(self, lr, momentum, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.cnn = EncoderCNN()
        self.fc = nn.Linear(IMG_ENCODE_SIZE, N_CLASSES)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.train_acc = Accuracy('multiclass', num_classes=N_CLASSES)
        self.val_acc = Accuracy('multiclass', num_classes=N_CLASSES)

    def forward(self, x, y):
        batch_size = len(x)
        x = self.cnn(x).view(batch_size, -1)
        y_pred = self.fc(x)
        return y_pred, y

    def training_step(self, batch, batch_idx):
        y_pred, y = self(*batch)
        loss = F.cross_entropy(y_pred, y)
        self.train_acc.update(y_pred, y)
        return loss

    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        y_pred, y = self(*batch)
        loss = F.cross_entropy(y_pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.val_acc.update(y_pred, y)

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute())

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        return optimizer, scheduler