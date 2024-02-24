""" Train the CNN model using dataset
"""

# python RNNTrain_chiquitto.py --pos ../dataset/pos.csv --neg ../dataset/neg.csv --output models

import sys
sys.path.append("../data") 
from RNNModel import RNN_model
import dataSetPartition
import keras
import os

from RNNTrain_args import process_argv

opts = process_argv()
print(f"opts={opts}")

def RNN_train(x_dataset,y_dataset,lerr,dr):
    model = RNN_model(lerr,dr)
    '''
    # transfer learning
    if os.path.exists("RNN_model_preTrained.h5"):
        print("load the weights")
        model.load_weights("RNN_model_preTrained.h5")
    '''        
    model.fit(x_dataset,y_dataset,batch_size = 128, epochs = 128,\
          validation_split = 0.2)
    print("model train over")
    return model

if __name__ == "__main__":
    positive = opts['pos']
    negative = opts['neg']
    x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset = \
      dataSetPartition.train_test_partition(positive,negative,doshuffle=True)

    ler = [0.0004] # learning rate
    drp = [0.3]

    # Applying grid search we obtained the following
    # values: learning rate = 0.0004,
    # dropout probability = 0.3, recurrent dropout = 0.3
    model = RNN_train(x_train_dataset,y_train_dataset, lerr=ler[0], dr=drp[0])

    modelfile = f"{opts['output']}/RNN_model.h5"
    model.save(modelfile)
    print(f"The model is saved as {modelfile}")
