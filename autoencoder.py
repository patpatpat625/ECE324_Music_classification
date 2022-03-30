import os

import torch
import torchvision
from torch import nn


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding='same'),
            nn.UpSampling2D(scale_factor=2, mode='bilinear'),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same'),
            nn.UpSampling2D(scale_factor=2, mode='bilinear'),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding='same'),
            nn.UpSampling2D(scale_factor=2, mode='bilinear'))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    # load training, validation, and testing
    dir = "D:\\music_classifier\\data\\"
    x1_train = np.load(dir+"train\\spectrogram.npy")
    x2_train = np.load(dir + "train\\feature.npy")
    y_train = np.load(dir+"train\\label_num.npy")

    # define parameters
    epochs = 20
    lr = 0.0005
    batch = 1

    # Initialize model
    model = autoencoder()

    # Initialize loss function and optimizer
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Training
    for epoch in range(epochs):
        start = 0
        accuracy = 0
        val_accuracy = 0
        while start < len(x1_train):
            # get a new training data
            curr_x1_train = torch.from_numpy(x1_train[start:start + batch, :, :]).unsqueeze(1).float()
            curr_x2_train = torch.from_numpy(x2_train[start:start + batch, :, :]).unsqueeze(1).float()
            curr_y_train = torch.tensor(y_train[start:start + batch]).type(torch.LongTensor)
            # increase start index by batch size
            start += batch

            y_pred = model(curr_x1_train, curr_x2_train)

            accuracy += (y_pred.argmax(axis=1) == curr_y_train).sum()
            curr_loss = loss(y_pred, curr_y_train)
            curr_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
