from Datasets.AIC2020 import DataGenerator
from Models.EfficientNet import EfficientNet 
from Metrics.ConfusionMatrix import ConfusionMatrix
import keras 
import numpy as np 

if __name__ == '__main__':
    batch_size = 64
    n_classes = 9
    ckpt_path = 'CheckPoint/EfficientNetB4.h5'
    model = keras.models.load_model(ckpt_path)
    model.summary()
    test_gen = DataGenerator('Data/image_train',
                          'list/val.csv',
                          batch_size = batch_size,
                          dim = (224, 224, 3),
                          n_classes = n_classes,
                          shuffle = False
                          )
    prediction = model.predict_generator(test_gen)
    pred_labels = np.argmax(prediction, axis=1)
    y_target =  test_gen.labels
    y_predict = pred_labels
    cm = ConfusionMatrix(n_classes)
    cm.calculate(target=y_target, predict=y_predict)
    print("Accuracy: ", cm.calc_acc())
    cm.display('cm.png', type = 'rec')

    