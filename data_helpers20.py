import os
import numpy as np
import re
from keras.utils.np_utils import to_categorical


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\\n", " ", string)
#   string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(test_or_train, categories):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    x_text = []
    labels= []
    for i in range(len(categories)):
        path = "20news-bydate-" + test_or_train + '/' + categories[i]
        for file_name in os.listdir(path):
            s = str(open(path+"/"+file_name, "rb").read())
            s = clean_str(s)
            words = s.split(" ")
            if len(words) > 1500:
                words = words[:1500]
                s = " ".join(words)
            x_text.append(s)
            labels.append(i)

    """
    lengths = {}
    for i in range(len(categories)):
        path = "20news-bydate-train/" + categories[i]
        for file_name in os.listdir(path):
            s = str(open(path+"/"+file_name, "rb").read())
            s = clean_str(s)
            num_of_words = len(s.split(" "))
            lengths[num_of_words] = lengths.get(num_of_words,0) + 1

    print(lengths)
    print(sum(lengths.keys()))
    """
    # Load data from files
    # Split by words
#    clean_str(x_text[-1])
    x_text = [clean_str(sent) for sent in x_text]
#   print(x_text[-1])
    # Generate labels
    y = to_categorical(np.asarray(labels))
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
