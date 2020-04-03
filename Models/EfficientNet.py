#Build Model 
#import tensorflow as tf
from keras.layers import Dense 
from keras.models import Model
import efficientnet.keras as efn 
from keras.applications.resnet import ResNet101

def EfficientNet(n_classes, unfreeze = True):
    base_model = efn.EfficientNetB4(weights = 'imagenet', include_top = False, pooling = 'avg')
    for layer in base_model.layers:
        layer.trainable = unfreeze
    out = base_model.output
    out = Dense(n_classes, activation = 'softmax')(out)
    model = Model(base_model.input, out)
    return model
