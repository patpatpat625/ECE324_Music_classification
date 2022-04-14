import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# model definition
# Using CNNs
class CNN_Ensemble(nn.Module):
    def __init__(self):
        # call super to initialize the class above in the hierarchy
        super(CNN_Ensemble, self).__init__()

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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding='same')

        self.conv1.weight.data.uniform_(0, 0.05)
        self.conv2.weight.data.uniform_(0, 0.05)
        self.conv3.weight.data.uniform_(0, 0.05)

        self.network4 = torch.nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=3),

            self.conv2,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=8),

            self.conv3,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=3))
        
        # combined
        self.linear1 = nn.Linear(17904, 8)


    def forward(self, x1, x2):

        three = self.network3(x1)
        four = self.network4(x2)

        # flatten and concat the two matrices
        x = torch.cat((torch.flatten(three, start_dim=1), torch.flatten(four, start_dim=1)), dim=1)
        return torch.sigmoid(self.linear1(x))


def plot(title, y):
    plt.title("Training and Validation Accuracy")
    plt.plot(y)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.show()


if __name__ == "__main__":
    device = 'cpu'
    # load data and convert to tensor
    dir = "C:\\Users\\xtzha\\Desktop\\ECE324\\ECE324_Music_classification\\data\\"

    x1_train = np.load(dir+"train10\\spectrogram.npy")
    x1_train = torch.from_numpy(x1_train).unsqueeze(1).float().to(device)
    x2_train = np.load(dir + "train10\\feature.npy")
    x2_train = torch.from_numpy(x2_train).unsqueeze(1).float().to(device)
    y_train = np.load(dir+"train10\\label.npy")
    y_train = torch.tensor(y_train).type(torch.LongTensor).to(device)

    x1_val = np.load(dir + "val10\\spectrogram.npy")
    x1_val = torch.from_numpy(x1_val).unsqueeze(1).float().to(device)
    x2_val = np.load(dir + "val10\\feature.npy")
    x2_val = torch.from_numpy(x2_val).unsqueeze(1).float().to(device)
    y_val = np.load(dir + "val10\\label.npy")
    y_val = torch.tensor(y_val).type(torch.LongTensor).to(device)

    x1_test = np.load(dir + "test10\\spectrogram.npy")
    x1_test = torch.from_numpy(x1_test).unsqueeze(1).float().to(device)
    x2_test = np.load(dir + "test10\\feature.npy")
    x2_test = torch.from_numpy(x2_test).unsqueeze(1).float().to(device)
    y_test = np.load(dir + "test10\\label.npy")
    y_test = torch.tensor(y_test).type(torch.LongTensor).to(device)

    # define parameters
    epochs = 20
    lr = 0.0001
    batch = 128

    # Initialize model
    model = CNN_Ensemble().to(device)

    # Initialize loss function and optimizer
    loss = nn.CrossEntropyLoss()
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

