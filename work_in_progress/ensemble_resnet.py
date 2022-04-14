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
        self.network1 = torch.nn.Sequential(*(list(models.resnet50(pretrained=True).children())[:-1]))

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
    device = 'cuda'
    np.random.seed(65)
    # load data and convert to tensor
    dir = "D:\\music_classifier\\data_10\\"

    x1_train = np.load(dir+"train\\spectrogram.npy")
    x1_train = torch.from_numpy(x1_train).unsqueeze(1).repeat(1, 3, 1, 1).float()
    x2_train = np.load(dir + "train\\feature.npy")
    x2_train = torch.from_numpy(x2_train).unsqueeze(1).float()
    y_train = np.load(dir+"train\\label.npy")
    y_train = torch.tensor(y_train).type(torch.LongTensor)

    x1_val = np.load(dir + "val\\spectrogram.npy")
    x1_val = torch.from_numpy(x1_val).unsqueeze(1).repeat(1, 3, 1, 1).float()
    x2_val = np.load(dir + "val\\feature.npy")
    x2_val = torch.from_numpy(x2_val).unsqueeze(1).float()
    y_val = np.load(dir + "val\\label.npy")
    y_val = torch.tensor(y_val).type(torch.LongTensor)

    # define parameters
    epochs = 20
    lr = 0.0005
    batch = 100
    # Initialize model
    model = Resnet_Ensemble()

    # Initialize loss function and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-6)
    # index array
    indices = np.arange(x1_train.shape[0])

    # Training
    for epoch in range(epochs):
        start = 0
        accuracy = 0
        val_accuracy = 0
        # shuffle the indices to remove correlation
        np.random.shuffle(indices)
        while start < len(x1_train):
            end = start + batch
            if end > len(y_train):
                end = len(y_train)
            curr = indices[start:end]
            # get a new training data
            curr = indices[start:start + batch]
            curr_x1_train = x1_train[curr, :, :]
            curr_x2_train = x2_train[curr, :, :]
            curr_y_train = y_train[curr]
            # increase start index by batch size
            start += batch

            y_pred = model(curr_x1_train, curr_x2_train)

            accuracy += (y_pred.argmax(axis=1) == curr_y_train).sum()
            curr_loss = loss(y_pred, curr_y_train)
            curr_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # calculate validation accuracy
            start = 0
            while start < len(x1_val):
                # get a new training data
                curr_x1_val = x1_val[start:start + batch, :, :]
                curr_x2_val = x2_val[start:start + batch, :, :]
                curr_y_val = y_val[start:start + batch]
                # increase start index by batch size
                start += batch

                y_pred_val = model(curr_x1_val, curr_x2_val)

                val_accuracy += (y_pred_val.argmax(axis=1) == curr_y_val).sum()

            print('epoch:', epoch + 1)
            print('\t correct items', accuracy.item())
            print('\t training accuracy =', round(accuracy.item() / len(x1_train) * 100, 4))
            print('\t validation accuracy =', round(val_accuracy.item() / len(x1_val) * 100, 4))
