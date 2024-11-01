from keras.models import load_model
from keras.models import Sequential
import tensorflow.keras.backend as K
import numpy as np

model_path = 'models/CNN_model.h5'
model_path = 'models/RNN_model.h5'

model = load_model(model_path)

trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

print('Total params: {:,}'.format(trainable_count + non_trainable_count))
print('Trainable params: {:,}'.format(trainable_count))
print('Non-trainable params: {:,}'.format(non_trainable_count))
