import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display


def pie_graph(y, mylabels):

    plt.pie(y, labels=mylabels, autopct='%1.1f%%')
    plt.title("Percentage of Data of Each Composer in Training Set")
    plt.show()


def plot_training_dist():
    # load data
    results = np.zeros(8)
    dir = "D:\\music_classifier\\data\\"
    y_train = np.load(dir + "train\\label.npy")
    for data in y_train:
        results += data

    pie_graph(results, ["Bach", "Vivaldi", "Mozart", "Beethoven", "Debussy", "Chopin", "Brahms", "Tchaikovsky"])


def plot_spectrogram():
    # load data
    dir = "D:\\music_classifier\\data\\"
    x_train = np.load(dir + "train\\spectrogram.npy")
    S = x_train[0]
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=22050, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


if __name__ == "__main__":
    plot_spectrogram()
