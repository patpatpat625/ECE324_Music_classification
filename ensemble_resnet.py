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
        self.network1 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        # change input channel to 1
        self.network1.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3,bias=False)

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
        self.linear = nn.Linear(3544, 8)

    def forward(self, x1, x2):
        one = self.network1(x1)
        two = self.network2(x2)

        # flatten and concat the two matrices
        x = torch.cat((torch.flatten(one), torch.flatten(two)), dim=0)
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

    # define parameters
    epochs = 20
    lr = 0.0005
    batch = 1
    # Initialize model
  #  resnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
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
            curr_x1_train = torch.from_numpy(x1_train[start, :, :]).unsqueeze(0).unsqueeze(0).float()
            curr_x2_train = torch.from_numpy(x2_train[start, :, :]).unsqueeze(0).float()
            curr_y_train = torch.tensor(y_train[start]).type(torch.LongTensor)
            start += 1

       #     print(curr_x1_train.shape)

            y_pred = model(curr_x1_train, curr_x2_train)

            if (y_pred.argmax() == curr_y_train):
                accuracy += 1
            curr_loss = loss(y_pred, curr_y_train)
            curr_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('epoch:', epoch + 1,  'accuracy =', accuracy, ' ', accuracy / len(x1_train))
