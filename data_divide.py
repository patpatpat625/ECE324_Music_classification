import numpy as np


def save_to_csv(filename, arr):
    np.save(filename, arr)

def append_to(destination, input):
    for data in input:
        destination.append(data)
    return destination

if __name__ == "__main__":
    dir = 'D:\\music_classifier\\data_10\\'
    sub_dir = {"Bach\\", "Vivaldi\\", "Mozart\\", "Beethoven\\", "Brahms\\", "Chopin\\", "Debussy\\", "Tchaikovsky\\"}
    # define where to store the data
    x1_train = []
    x1_val = []
    x1_test = []
    x2_train = []
    x2_val = []
    x2_test = []
    y_train = []
    y_val = []
    y_test = []

    for folder in sub_dir:
        x1 = np.load(dir + folder + "spectrogram.npy", allow_pickle=True)
        x2 = np.load(dir + folder + "feature.npy", allow_pickle=True)
        y = np.load(dir + folder + "label.npy", allow_pickle=True)
        total_length = 0
        for piece in y:
            labels = []
            total_length += len(piece)
        # estimate train, val, and test index
        train = 0.6*total_length
        val = 0.8*total_length
        # keep track of current size
        curr_size = 0
        for i in range(len(y)):
            size = len(y[i])
            # if this should go into test set
            if curr_size + size > val:
                x1_test = append_to(x1_test, x1[i])
                x2_test = append_to(x2_test, x2[i])
                y_test = append_to(y_test, y[i])
            elif curr_size + size > train:
                x1_val = append_to(x1_val, x1[i])
                x2_val = append_to(x2_val, x2[i])
                y_val = append_to(y_val, y[i])
            else:
                x1_train = append_to(x1_train, x1[i])
                x2_train = append_to(x2_train, x2[i])
                y_train = append_to(y_train, y[i])
            curr_size += size

    # convert to np arrays and save to file
    save_to_csv(dir + "train\\spectrogram", x1_train)
    save_to_csv(dir + "train\\feature", x2_train)
    save_to_csv(dir + "train\\label", y_train)
    save_to_csv(dir + "val\\spectrogram", x1_val)
    save_to_csv(dir + "val\\feature", x2_val)
    save_to_csv(dir + "val\\label", y_val)
    save_to_csv(dir + "test\\spectrogram", x1_test)
    save_to_csv(dir + "test\\feature", x2_test)
    save_to_csv(dir + "test\\label", y_test)
