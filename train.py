import s2s
import mystr
import myfile
import numpy as np
import pickle
from tqdm import *
from numpy import array
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

MAX_INPUT = 70
MAX_OUTPUT = 20000
WORD_DIM = 195102

csvArr = myfile.readcsv('./datasets/news20171225.csv')

inputArr = []
outputArr = []
X = np.array([]).reshape(-1, MAX_INPUT)
Y = np.array([]).reshape(-1, MAX_OUTPUT, WORD_DIM)

for i in range(0, len(csvArr)):
    inputArr.append(csvArr[i][0])
    outputArr.append(csvArr[i][2])

with open('label.pickle', 'rb') as f:
    encoder = pickle.load(f)

    for i in range(0, len(inputArr)):
        print()
        raw = array(encoder.transform(list(inputArr[i])))
        fill = np.append(raw, np.zeros(MAX_INPUT - raw.size)).reshape(-1, MAX_INPUT)
        X = np.append(X, fill, axis=0)
        
        raw = array(encoder.transform(list(inputArr[i])))
        fill = np.append(raw, np.zeros(MAX_OUTPUT - raw.size)).reshape(-1, MAX_OUTPUT)
        with open('onehot.pickle', 'rb') as f2:
            onehot = pickle.load(f2)
            val=onehot.transform(fill.reshape(fill.size, 1)) # csr_matrix