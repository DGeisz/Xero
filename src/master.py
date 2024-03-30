import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np


training_data = datasets.MNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


def draw_weights(w, Kx, Ky, s_len, fig):
    tapestry = np.zeros((s_len * Ky, s_len * Kx))

    w_i = 0
    for y in range(Ky):
        for x in range(Kx):
            tapestry[y * s_len : (y + 1) * s_len, x * s_len : (x + 1) * s_len] = w[
                w_i
            ].reshape(s_len, s_len)
            w_i += 1

    plt.clf()
    max_val = np.max(np.abs(tapestry))
    im = plt.imshow(tapestry, cmap="bwr", vmax=max_val, vmin=-max_val)
    fig.colorbar(im, ticks=[0, max_val])
    plt.axis("off")
    fig.canvas.draw()


class BasicModel(nn.Module):
    dim_x: int
    dim_y: int

    def __init__(self, dim_x=10, dim_y=10):
        super().__init__()

        self.dim_x = dim_x
        self.dim_y = dim_y

        total_neurons = dim_x * dim_y

        self.flatten = nn.Flatten()
        self.layer_1 = nn.Sequential(nn.Linear(28 * 28, total_neurons), nn.ReLU())
        self.layer_2 = nn.Sequential(nn.Linear(total_neurons, 10), nn.Softmax())

        self.init_first_layer = self.layer_1[0].weight.data.clone()

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        return x

    def draw_first_layer(self):
        fig = plt.figure(figsize=(10, 10))
        draw_weights(self.layer_1[0].weight.data, self.dim_x, self.dim_y, 28, fig)

    def draw_first_layer_diff(self):
        fig = plt.figure(figsize=(10, 10))
        draw_weights(
            self.layer_1[0].weight.data - self.init_first_layer,
            self.dim_x,
            self.dim_y,
            28,
            fig,
        )

    def draw_data_overlay(self, data):
        fig = plt.figure(figsize=(10, 10))
        overlay = self.layer_1[0].weight * torch.flatten(data)
        together = (4 * overlay) + self.layer_1[0].weight

        draw_weights(
            together.detach().numpy(),
            self.dim_x,
            self.dim_y,
            28,
            fig,
        )


class ModelTrainer:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        train_loader,
        test_loader,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch, (X, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if (batch + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch+1}, Loss: {loss.item()}")

            self.test()

    def test(self):
        self.model.eval()
        total, correct = 0, 0

        with torch.no_grad():
            for X, y in self.test_loader:
                pred = self.model(X)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                total += y.size(0)
        print(f"Test Accuracy: {correct / total * 100}%")
