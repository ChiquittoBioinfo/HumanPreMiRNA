""" Construct the RNN model for deep learning
"""

import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.layers import LSTM,Masking,Embedding
from keras.optimizers import Adam

def RNN_model(lerr,dr):
    SEG_LENTH = 180
    model = Sequential()
    model.add(Masking(mask_value= [0,0,0,0,0,0,0,0,0,0,0,0],\
             input_shape=(SEG_LENTH, 12)))
    model.add(LSTM(128,dropout=dr, recurrent_dropout=dr,\
                   #kernel_regularizer = regularizers.l2(0.1),\
                   input_shape = (SEG_LENTH, 12),return_sequences = True))
    model.add(LSTM(128,dropout=dr, recurrent_dropout=dr,\
                   #kernel_regularizer = regularizers.l2(0.1),\
                   return_sequences = True))
    '''
    model.add(LSTM(64,dropout=dr, recurrent_dropout=dr,\
              # kernel_regularizer = regularizers.l2(0.1),
               return_sequences = True))
    '''
    model.add(LSTM(64,dropout=dr, recurrent_dropout=dr,\
               #kernel_regularizer = regularizers.l2(0.1),\
               return_sequences = True))
    
    #128 128 64 2 2ds
    model.add(LSTM(2))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    
    adam = Adam(lr=lerr)
    model.compile(loss = 'categorical_crossentropy',optimizer = adam,\
                  metrics = ['accuracy'])
    print(model.summary())
    return model 

if __name__ == "__main__":
     model = RNN_model()
