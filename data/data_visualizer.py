import matplotlib.pyplot as plt
import numpy as np


def pie_graph(y, mylabels):

    plt.pie(y, labels=mylabels, autopct='%1.1f%%')
    plt.title("Percentage of Data of Each Composer in Training Set")
    plt.show()

if __name__ == "__main__":
    # load data
    '''
    results = np.zeros(8)
    dir = "D:\\music_classifier\\data\\"
    y_train = np.load(dir + "train\\label.npy")
    for data in y_train:
        results += data
    '''
    results = [7.8, 7.3, 5.8, 13.7, 8.4, 6.9, 19.2, 11.8]
    pie_graph(results, ["Bach", "Vivaldi", "Mozart", "Beethoven", "Debussy", "Chopin", "Brahms", "Tchaikovsky"])

