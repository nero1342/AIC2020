
from pathlib import Path
from PIL import Image
import numpy as np 
import pandas as pd 
import os
import keras
import timeit

class DataGenerator(keras.utils.Sequence):
    """
    """
    def __init__(self, root_dir, csv_path, batch_size = 32, dim = (32,32,32), n_classes = 10, shuffle = True):
        self.root_dir = Path(root_dir)
        self.dim = dim 
        self.df = pd.read_csv(csv_path, dtype = {'img_id': str})
        self.list_IDs = self.df['img_id']
        self.labels = self.df['vehicle_type']
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.on_epoch_end() 
    def __len__(self):
        return int((len(self.list_IDs) - 1) / self.batch_size) + 1

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:
          min(len(self.list_IDs), (index + 1) * self.batch_size)]
        list_IDs_temp = [k for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def read_image(self, image_path, size = 224):
        """
        Đọc và convert ảnh về (224, 224, 3)
        """
        image = Image.open(image_path)
        image = image.convert('RGB').resize((size, size), Image.ANTIALIAS)
        return np.asarray(image) / 255.

    def __data_generation(self, list_IDs_temp):
        X = np.empty((len(list_IDs_temp), *self.dim), dtype = 'float16')
        y = np.empty(len(list_IDs_temp), dtype = np.uint8)
        for i, ID in enumerate(list_IDs_temp):
            img_path = self.root_dir / (str(self.list_IDs[ID]) + '.jpg')
            X[i,] = self.read_image(img_path)
            y[i] = self.labels[ID]
        #print(list_IDs_temp[0])
        return X, y
