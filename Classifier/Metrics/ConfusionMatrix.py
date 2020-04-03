from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

class ConfusionMatrix():
    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.reset()
    def calculate(self, target, predict):
        # Confusion matrix
        self.cm = confusion_matrix(target, predict, range(self.nclasses))
        self.acc = self.cm.trace() / target.shape[0]
        # Precision - Recall
        true_array = np.zeros(self.nclasses)
        pos_array = np.zeros(self.nclasses)
        for i in range(self.nclasses):
            for j in range(self.nclasses):
                true_array[i] += self.cm[i][j]
                pos_array[j] += self.cm[i][j]

        for i in range(self.nclasses):
            for j in range(self.nclasses):
                self.pre[i][j] = self.cm[i][j] / pos_array[j]
                self.rec[i][j] = self.cm[i][j] / true_array[i]
        
        return self.cm 

    def calc_acc(self):
        return self.acc

    def reset(self):
        self.cm = np.zeros(shape=(self.nclasses, self.nclasses))
        self.pre = np.zeros(shape=(self.nclasses, self.nclasses))
        self.rec = np.zeros(shape=(self.nclasses, self.nclasses))

    def value(self, i, j):
        return self.cm[i][j]

    def summary(self):
        return self.cm

    def display(self, save_name = '', type = 'cm'):
        if type == 'pre':
            data_to_display = self.pre
        elif type == 'rec':
            data_to_display = self.rec
        else:
            data_to_display = self.cm 
        df_cm = pd.DataFrame(data_to_display, index=range(self.nclasses), columns=range(self.nclasses))
        fig = plt.figure(figsize = (20,14))
        fig.suptitle(type)
        sn.heatmap(df_cm, annot=True, cmap='YlGnBu', fmt = 'g')
        plt.tight_layout()
        if save_name != '':
            plt.savefig(save_name)
        plt.show()
        plt.close()
