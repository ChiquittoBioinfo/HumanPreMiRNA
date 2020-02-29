""" Construct the CNN model for deep learning
"""

import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten
from keras.layers import Conv1D,MaxPooling1D
from keras.optimizers import Adam

def CNN_model():

    model = Sequential()
    model.add(Conv1D(32,5,activation = 'relu',padding = 'same',\
              input_shape = (180,12)))
    model.add(MaxPooling1D(pool_size = 2))
    #first layer of convolution and max-pooling
    model.add(Conv1D(32,5,activation = 'relu',padding = 'same'))
    model.add(MaxPooling1D(pool_size = 2))
    #second layer of convolution and max-pooling
    model.add(Conv1D(64,6,activation = 'relu',padding = 'same'))
    model.add(MaxPooling1D(pool_size = 2))
    #third layer of convolution and max-pooling
    model.add(Conv1D(64,6,activation = 'relu',padding = 'same'))
    model.add(MaxPooling1D(pool_size = 2))
    
    model.add(Conv1D(128,7,activation = 'relu',padding = 'same'))
    model.add(MaxPooling1D(pool_size = 2))

    model.add(Flatten())
    model.add(Dropout(0.45))
    model.add(Dense(128,activation = 'relu',kernel_regularizer = regularizers.l2(0.01)))
    model.add(Dropout(0.45))
    model.add(Dense(2,activation = 'softmax'))
    adam = Adam(lr=0.0003)
    model.compile(loss = 'categorical_crossentropy',optimizer = adam,\
              metrics = ['accuracy'])
    print(model.summary())
    
    return model
if __name__ == "__main__":
     model = CNN_model()
