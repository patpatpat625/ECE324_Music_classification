import matplotlib.pyplot as plt
import numpy as np


def pie_graph(y, mylabels):

    plt.pie(y, labels=mylabels, autopct='%1.1f%%')
    plt.show()

if __name__ == "__main__":
    # load data
    results = np.zeros(8)
    dir = "D:\\music_classifier\\data\\"
    y_train = np.load(dir + "train\\label.npy")
    for row in y_train:
        results += row

    pie_graph(results, ["Bach", "Vivaldi", "Mozart", "Beethoven", "Debussy", "Chopin", "Brahms", "Tchaikovsky"])

