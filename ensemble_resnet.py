import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from tqdm import tqdm


# model definition
# Using CNNs
class Resnet_Ensemble(nn.Module):
    def __init__(self):
        # call super to initialize the class above in the hierarchy
        super(Resnet_Ensemble, self).__init__()

        # spectrogram CNN
        self.network1 = torch.nn.Sequential(*(list(models.resnet152(pretrained=True).children())[:-1]))

        # feature CNN
        self.network2 = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU())

        # combined
        self.linear = nn.Linear(4592, 8)

    def forward(self, x1, x2):
        one = self.network1(x1)
        two = self.network2(x2)

        # flatten and concat the two matrices
        x = torch.cat((torch.flatten(one, start_dim=1), torch.flatten(two, start_dim=1)), dim=1)
        return torch.sigmoid(self.linear(x))


def plot(title, y, y_label):
    plt.title(title)
    plt.plot(y)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.show()


if __name__ == "__main__":
    # load training, validation, and testing
    dir = "D:\\music_classifier\\data\\"
    x1_train = np.load(dir+"train\\spectrogram.npy")
    x2_train = np.load(dir + "train\\feature.npy")
    y_train = np.load(dir+"train\\label_num.npy")
    x1_val = np.load(dir + "validate\\spectrogram.npy")
    x2_val = np.load(dir + "validate\\feature.npy")
    y_val = np.load(dir + "validate\\label_num.npy")

    # define parameters
    epochs = 20
    lr = 0.0005
    batch = 120
    # Initialize model
    model = Resnet_Ensemble()

    # Initialize loss function and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Training
    for epoch in tqdm(range(epochs)):
        start = 0
        accuracy = 0
        val_accuracy = 0
        while start < len(x1_train):
            # get a new training data
            curr_x1_train = torch.from_numpy(x1_train[start:start + batch, :, :]).unsqueeze(1).repeat(1, 3, 1, 1).float()
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

        print('epoch:', epoch + 1,  'accuracy =', accuracy, ' ', accuracy / len(x1_train))
