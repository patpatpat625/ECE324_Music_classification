import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# model definition
# Using CNNs
class CNN_Ensemble1(nn.Module):
    def __init__(self):
        # call super to initialize the class above in the hierarchy
        super(CNN_Ensemble1, self).__init__()
        # spectrogram CNN
        self.network1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # feature CNN
        self.network2 = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # combined
        self.linear = nn.Linear(3392, 8)

    def forward(self, x1, x2):
        one = self.network1(x1)
        two = self.network2(x2)

        # flatten and concat the two matrices
        x = torch.cat((torch.flatten(one, start_dim=1), torch.flatten(two, start_dim=1)), dim=1)
        return torch.sigmoid(self.linear(x))


class CNN_Ensemble2(nn.Module):
    def __init__(self):
        # call super to initialize the class above in the hierarchy
        super(CNN_Ensemble2, self).__init__()

        # spectrogram CNN
        self.network3 = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=3),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=1))

        # feature CNN
        self.network4 = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=3),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=1))

        # combined
        self.linear = nn.Linear(3392, 8)

    def forward(self, x1, x2):

        three = self.network3(x1)
        four = self.network4(x2)

        # flatten and concat the two matrices
        x = torch.cat((torch.flatten(three, start_dim=1), torch.flatten(four, start_dim=1)), dim=1)
        return torch.sigmoid(self.linear(x))


class CNN_Ensemble3(nn.Module):
    def __init__(self):
        # call super to initialize the class above in the hierarchy
        super(CNN_Ensemble3, self).__init__()

        # spectrogram CNN
        self.network3 = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=3),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=3))

        # feature CNN
        self.network4 = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=3),
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=6),
            nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=3),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=1))

        # combined
        #self.linear = nn.Linear(5496, 8)
        self.linear = nn.Linear(10896, 8)

    def forward(self, x1, x2):

        three = self.network3(x1)
        four = self.network4(x2)

        # flatten and concat the two matrices
        x = torch.cat((torch.flatten(three, start_dim=1), torch.flatten(four, start_dim=1)), dim=1)
        return torch.sigmoid(self.linear(x))


def plot(title, y):
    plt.title("Training and Validation Accuracy")
    plt.plot(y)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.show()


if __name__ == "__main__":
    device = 'cpu'
    np.random.seed(555)
    # load data and convert to tensor
    dir = "D:\\music_classifier\\data_10\\"

    x1_train = np.load(dir+"train\\spectrogram.npy")
    x1_train = torch.from_numpy(x1_train).unsqueeze(1).float().to(device)
    x2_train = np.load(dir + "train\\feature.npy")
    x2_train = torch.from_numpy(x2_train).unsqueeze(1).float().to(device)
    y_train = np.load(dir+"train\\label.npy")
    y_train = torch.tensor(y_train).type(torch.LongTensor).to(device)

    x1_val = np.load(dir + "val\\spectrogram.npy")
    x1_val = torch.from_numpy(x1_val).unsqueeze(1).float().to(device)
    x2_val = np.load(dir + "val\\feature.npy")
    x2_val = torch.from_numpy(x2_val).unsqueeze(1).float().to(device)
    y_val = np.load(dir + "val\\label.npy")
    y_val = torch.tensor(y_val).type(torch.LongTensor).to(device)

    x1_test = np.load(dir + "test\\spectrogram.npy")
    x1_test = torch.from_numpy(x1_test).unsqueeze(1).float().to(device)
    x2_test = np.load(dir + "test\\feature.npy")
    x2_test = torch.from_numpy(x2_test).unsqueeze(1).float().to(device)
    y_test = np.load(dir + "test\\label.npy")
    y_test = torch.tensor(y_test).type(torch.LongTensor).to(device)

    # define parameters
    epochs = 20
    lr = 0.0001
    batch = 150

    # Initialize model
    model = CNN_Ensemble3().to(device)

    # Initialize loss function and optimizer
    loss = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr, momentum = 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    # index array
    indices = np.arange(x1_train.shape[0])

    # Training
    for epoch in range(epochs):
        start = 0
        accuracy = 0
        val_accuracy = 0
        test_accuracy = 0
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

        # calculate testing accuracy
        start = 0
        while start < len(y_test):
            # get a new testing data
            curr_x1_test = x1_test[start:start + batch, :, :]
            curr_x2_test = x2_test[start:start + batch, :, :]
            curr_y_test = y_test[start:start + batch]
            # increase start index by batch size
            start += batch

            y_pred_test = model(curr_x1_test, curr_x2_test)

            test_accuracy += (y_pred_test.argmax(axis=1) == curr_y_test).sum()

        print('epoch:', epoch + 1)
        print('\t correct items', accuracy.item())
        print('\t training accuracy =', round(accuracy.item() / len(y_train) * 100, 4))
        print('\t validation accuracy =', round(val_accuracy.item() / len(y_val) * 100, 4))
        print('\t testing accuracy =', round(test_accuracy.item() / len(y_test) * 100, 4))

