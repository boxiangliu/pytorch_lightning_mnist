import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.optim import Adam


class LitMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # MNIST images are (1, 28, 28) (channels, height, width)
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        log_prob = self(x)
        loss = F.nll_loss(log_prob, y)
        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        log_prob = self(x)
        loss = F.nll_loss(log_prob, y)
        self.log("dev_loss", loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # Download the train set:
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        # Download the test set:
        MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage):
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_train = MNIST(
            os.getcwd(), train=True, download=False, transform=transform
        )
        mnist_test = MNIST(
            os.getcwd(), train=False, download=False, transform=transform
        )

        mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

dm = MNISTDataModule()
model = LitMNIST()
wandb_logger = WandbLogger()
trainer = Trainer(
    gpus=8,
    strategy="ddp",
    logger=wandb_logger,
    log_every_n_steps=10,
    flush_logs_every_n_steps=10,
)
trainer.fit(model, dm)
