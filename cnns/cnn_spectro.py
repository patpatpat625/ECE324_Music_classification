import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class CNN_feature(nn.Module):
    def __init__(self):
        # call super to initialize the class above in the hierarchy
        super(CNN_feature, self).__init__()

        self.network = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=3),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=1))

        # combined
        self.linear1 = nn.Linear(6880, 1000)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1000, 8)
        # self.linear = nn.Linear(5184, 8)

    def forward(self, x):
        x = self.network(x)
        # flatten and concat the two matrices
        x = torch.flatten(x, start_dim=1)
        return torch.sigmoid(self.linear2(self.relu(self.linear1(x))))


if __name__ == "__main__":
    np.random.seed(829)
    # load data and convert to tensor
    dir = "D:\\music_classifier\\data_20\\"

    x1_train = np.load(dir+"train\\spectrogram.npy")
    x1_train = torch.from_numpy(x1_train).unsqueeze(1).float()
    y_train = np.load(dir+"train\\label.npy")
    y_train = torch.tensor(y_train).type(torch.LongTensor)

    x1_val = np.load(dir + "val\\spectrogram.npy")
    x1_val = torch.from_numpy(x1_val).unsqueeze(1).float()
    y_val = np.load(dir + "val\\label.npy")
    y_val = torch.tensor(y_val).type(torch.LongTensor)

    x1_test= np.load(dir + "test\\spectrogram.npy")
    x1_test = torch.from_numpy(x1_test).unsqueeze(1).float()
    y_test = np.load(dir + "test\\label.npy")
    y_test = torch.tensor(y_test).type(torch.LongTensor)

    # define parameters
    epochs = 30
    lr = 0.0001
    batch = 150

    # Initialize model
    model = CNN_feature()

    # Initialize loss function and optimizer
    loss = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr, momentum = 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    # index array
    indices = np.arange(y_train.shape[0])

    # Training
    for epoch in range(epochs):
        start = 0
        accuracy = 0
        val_accuracy = 0
        test_accuracy = 0
        # shuffle the indices to remove correlation
        np.random.shuffle(indices)
        while start < len(y_train):
            # get a new training data
            end = start + batch
            if end > len(y_train):
                end = len(y_train)
            curr = indices[start:end]
            curr_x1_train = x1_train[curr, :, :]
            curr_y_train = y_train[curr]
            # increase start index by batch size
            start += batch

            y_pred = model(curr_x1_train)

            accuracy += (y_pred.argmax(axis=1) == curr_y_train).sum()
            curr_loss = loss(y_pred, curr_y_train)
            curr_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # calculate validation accuracy
        start = 0
        while start < len(y_val):
            # get a new training data
            curr_x1_val = x1_val[start:start + batch, :, :]
            curr_y_val = y_val[start:start + batch]
            # increase start index by batch size
            start += batch

            y_pred_val = model(curr_x1_val)

            val_accuracy += (y_pred_val.argmax(axis=1) == curr_y_val).sum()

        # calculate testing accuracy
        start = 0
        while start < len(y_test):
            # get a new training data
            curr_x1_test = x1_test[start:start + batch, :, :]
            curr_y_test = y_test[start:start + batch]
            # increase start index by batch size
            start += batch

            y_pred_test = model(curr_x1_test)

            test_accuracy += (y_pred_test.argmax(axis=1) == curr_y_test).sum()

        print('epoch:', epoch + 1)
        print('\t correct items', accuracy.item())
        print('\t training accuracy =', round(accuracy.item() / len(y_train) * 100, 4))
        print('\t validation accuracy =', round(val_accuracy.item() / len(y_val) * 100, 4))
        print('\t testing accuracy =', round(test_accuracy.item() / len(y_test) * 100, 4))
    # save model
    #torch.save(model.state_dict(), "C:\\Users\\patty\\Desktop\\ECE324_Music_classification\\'s_model.pth'")