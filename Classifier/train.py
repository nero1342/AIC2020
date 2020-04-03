from Datasets.AIC2020 import DataGenerator
from Models.EfficientNet import EfficientNet 
import keras 

def train(epochs, learning_rate, batch_size, n_classes, ckpt_path):
    epochs = epochs
    learning_rate = learning_rate
    batch_size = batch_size
    n_classes = n_classes
    ckpt_path = ckpt_path
    mcp = keras.callbacks.ModelCheckpoint(ckpt_path,monitor = 'val_acc', save_best_only=True)
    model = EfficientNet(n_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate), 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    train_gen = DataGenerator('Data/image_train',
                          'list/train.csv',
                          batch_size = batch_size,
                          dim = (224, 224, 3),
                          n_classes = n_classes,
                          shuffle = True
                          )
    val_gen = DataGenerator('Data/image_train',
                          'list/val.csv',
                          batch_size = batch_size,
                          dim = (224, 224, 3),
                          n_classes = n_classes,
                          shuffle = True
                          )
    history = model.fit_generator(
            generator=train_gen,
            epochs = epochs, 
            verbose = 1, 
            callbacks = [mcp],
            validation_data = val_gen
    )
    return history
    
if __name__ == '__main__':
    train(
        epochs = 4,
        learning_rate = 0.0001,
        batch_size = 16,
        n_classes = 9,
        ckpt_path = 'CheckPoint/EfficientNetB4.h5'
    )