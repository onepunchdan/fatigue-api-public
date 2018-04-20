import numpy as np


class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, seq_dim = 11, seq_len = 200, batch_size = 20, shuffle = False):
        'Initialization'
        self.seq_dim = seq_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, labels, list_IDs):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

            # Generate data
            X, y = self.__data_generation(labels, list_IDs_temp)

            yield X, y

    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, labels, list_IDs_temp):
        'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.seq_len, self.seq_dim))
        y = np.empty((self.batch_size), dtype = int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store volume
            X[i, :, :] = np.load(ID)[:,:11]

            # Store class
            y[i] = labels[ID]
        return X, sparsify(y)

def sparsify(y):
    'Returns labels in binary NumPy array'
    n_classes = 2# Enter number of classes
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                   for i in range(y.shape[0])])