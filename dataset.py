from scipy import sparse
import numpy as np
import pickle

class DataSet():
    """
    Utility class to handle dataset structure.
    """

    def __init__(self, dataset, uselog=False):
        self._dataset = dataset
        self._epochs_completed = {}
        self._epochs_completed['train'] = 0
        self._epochs_completed['val'] = 0
        self._index_in_epoch = {}
        self._index_in_epoch['train'] = 0      
        self._index_in_epoch['val'] = 0
        self._num_samples = {}
        self._num_samples['train'] = len(self._dataset['train']['X'])
        self._num_samples['val'] = len(self._dataset['val']['X'])
        # Shuffle train dataset
        perm = np.arange(self._num_samples['train'])
        np.random.shuffle(perm)
        self._dataset['train']['X'] = [ self._dataset['train']['X'][i] for i in perm]
        self._dataset['train']['Y'] = [ self._dataset['train']['Y'][i] for i in perm]
        self._dataset['train']['ncodpers'] = [ self._dataset['train']['ncodpers'][i] for i in perm]
        
        
    def next_batch(self, batch_size, set_name='train'):
        start = self._index_in_epoch[set_name]
        self._index_in_epoch[set_name] += batch_size
        if self._index_in_epoch[set_name] > self._num_samples[set_name]:
            self._epochs_completed[set_name] += 1
            
            
            perm = np.arange(self._num_samples[set_name])
            np.random.shuffle(perm)
            
            self._dataset[set_name]['X'] = [ self._dataset[set_name]['X'][i] for i in perm]
            self._dataset[set_name]['Y'] = [ self._dataset[set_name]['Y'][i] for i in perm]
            self._dataset[set_name]['ncodpers'] = [ self._dataset[set_name]['ncodpers'][i] for i in perm]
            
            start = 0
            self._index_in_epoch[set_name] = batch_size
            assert batch_size <= self._num_samples[set_name]
            
        end = self._index_in_epoch[set_name]
        X = self._dataset[set_name]['X'][start:end]
        Y = self._dataset[set_name]['Y'][start:end]
        
        batch_x = []
        batch_y = []
        for i in range(len(X)):
            batch_x.append(X[i].toarray())
            batch_y.append(Y[i].toarray())
            
        # Convert lists to np.array
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        # Reshape times for the correct shape in tensorflow
        batch_y = batch_y.reshape((batch_y.shape[0], batch_y.shape[2]))
            
        return batch_x, batch_y
        
    def get_current_epoch(self, set_name):
        return self._epochs_completed[set_name]
        
    def get_set(self, set_name):
        X = []
        Y = []
        for i in range(len(self._dataset[set_name]['X'])):
            X.append(self._dataset[set_name]['X'][i].toarray())
            Y.append(self._dataset[set_name]['Y'][i].toarray())
        
            
        # Convert lists to np.array
        X = np.array(X)
        Y = np.array(Y)
        
        # Reshape times for the correct shape in tensorflow
        Y = Y.reshape((Y.shape[0], Y.shape[2]))
            
        return X, Y
        
       
    
