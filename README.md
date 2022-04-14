# ECE324_Music_classification

## Table of Contents

- [Introduction](https://github.com/patpatpat625/ECE324_Music_classification/edit/main/README.md#introduction)
- [Respository Setup](https://github.com/patpatpat625/ECE324_Music_classification/edit/main/README.md#repository-setup)
- [Data](https://github.com/patpatpat625/ECE324_Music_classification/edit/main/README.md#data)

<a name="Introduction"></a>
## Introduction
This repository contains 3 proposed Convolutional Neural Networks (CNNs) that aims to classify classical music composers based on data from International Music Score Library Project (IMSLP). The first CNN accepts spectrogram of the audio clip as inputs. The second CNN extracts various features from the audio clip using Python's library Librosa and pass these numpy arrays in as inputs. The last CNN is an ensemble as it performs classification with both spectrogram and features.

<a name="Repository Setup"></a>
## Repository Setup
### `cnns`
  
  contains the initialization and training loop for all the proposed CNNs
  
### `data` 

  should contain the training, validation and testing set. However, as these files are too large to upload, the folder is empty.


### `data_preprocessing`

  contains code for downloading data, converting them into arrays, train-test split and so on.
  
### `work_in_progress`

  contains 2 other models (Residual Network and Autoencoder) that could be implemented to further improve the classification accuracies. 

<a name="Data"></a>
## Data
As the dataset is too large to be put on github, please utilize the `data_preprocessing` folder to obtain the dataset. You could attain audio clips with various lengths through manipulating `sample_length` and `sample_ratio` in `data_convert.py`. You could also change the train-test split ratio in `data_divide.py` through `train` and `test`.
