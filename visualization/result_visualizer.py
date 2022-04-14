import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_conf_matrix(y_actual, y_pred, title):
    cf_matrix = confusion_matrix(y_actual, y_pred)

    ax = sns.heatmap(cf_matrix, annot=True, fmt='', cmap='Blues')

    ax.set_title(title);
    ax.set_xlabel('\nPredicted Composer')
    ax.set_ylabel('Actual Composer');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(["Bach", "Vivaldi", "Mozart", "Beethoven", "Debussy", "Chopin", "Brahms", "Tchaikovsky"])
    ax.yaxis.set_ticklabels(["Bach", "Vivaldi", "Mozart", "Beethoven", "Debussy", "Chopin", "Brahms", "Tchaikovsky"])

    ## Display the visualization of the Confusion Matrix.
    plt.show()


if __name__ == "__main__":
    pass
