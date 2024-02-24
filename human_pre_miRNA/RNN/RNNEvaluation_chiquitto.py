"""Evaluate the performance of the trained model using the test dataset
"""

# python RNNEvaluation_chiquitto.py --input ../dataset/pos.csv --model models/RNN_model.h5 --output models/output.csv

from keras.models import load_model
from keras.models import Sequential
import numpy as np
import math
import sys
sys.path.append("../data")
import dataSetPartition

import csv
from RNNEvaluation_args import process_argv

opts = process_argv()
print(f"opts={opts}")

def predict_class(model_path,x_test_dataset):
    print("load the model")
    try:
        model = load_model(model_path)
    except Exception:
        print("The model file doesn't exist!")
        exit(1)

    y_predict = model.predict(x_test_dataset)

    # y_cast = {True: [1,0],False:[0,1]}
    # Zero corresponds to Positive Class
    # One corresponds to Negative Class
    return np.argmax(y_predict,axis = 1)

if __name__ == "__main__":
    model_path = opts['model']

    positive = opts['input']
    negative = None
    x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset = \
          dataSetPartition.train_test_partition(positive,negative,doshuffle=False)
    
    y_predicted = predict_class(model_path,x_test_dataset)
    # print(y_predicted[:48])

    # Open CSV input
    csvinput = open(opts['input'], newline='')
    csvreader = csv.DictReader(csvinput, delimiter=',', quotechar='"')

    # Create CSV output
    fieldnames = ['id', 'class']
    csvoutput = opts['output']

    csvfile_writer = open(csvoutput, 'w', newline='')
    csvwriter = csv.DictWriter(csvfile_writer, fieldnames=fieldnames)
    csvwriter.writeheader()

    for n, row in enumerate(csvreader):
        csvwriter.writerow({ 'id': row['id'], 'class': y_predicted[n] })

    csvinput.close()
    csvfile_writer.close()
