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
from keras.callbacks import Callback, ProgbarLogger, BaseLogger

MAX_INPUT = 70
MAX_OUTPUT = 20000
WORD_DIM = 21574

csvArr = myfile.readcsv('./datasets/news20171225.csv')

inputArr = []
outputArr = []
X = np.array([]).reshape(-1, MAX_INPUT)

Ymatrix = []
Y = np.array([]).reshape(-1, MAX_OUTPUT, WORD_DIM)

print('讀取csv')
for i in trange(0, len(csvArr)):
    inputArr.append(csvArr[i][0])
    outputArr.append(csvArr[i][2])

f = open('label.pickle', 'rb')
encoder = pickle.load(f)

print('將X存放至Numpy, 將Y存放至csr_matrix')
for i in trange(0, len(inputArr)):
    raw = array(encoder.transform(list(inputArr[i])))
    fill = np.append(raw, np.zeros(MAX_INPUT - raw.size)
                     ).reshape(-1, MAX_INPUT)
    X = np.append(X, fill, axis=0)

    raw = array(encoder.transform(list(inputArr[i])))
    fill = np.append(raw, np.zeros(MAX_OUTPUT - raw.size)
                     ).reshape(-1, MAX_OUTPUT)

    with open('onehot.pickle', 'rb') as f2:
        onehot = pickle.load(f2)
        val = onehot.transform(fill.reshape(fill.size, 1))  # csr_matrix
        Ymatrix.append(val)

print(len(Ymatrix))

model = s2s.model1(MAX_INPUT, WORD_DIM, MAX_OUTPUT, 64)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])


def generator(Xnp, YSparse):
    print('開始Training...')
    for i in trange(0, len(YSparse)):
        yield Xnp[i].reshape(-1, MAX_INPUT), YSparse[i].toarray().reshape((-1, MAX_OUTPUT, WORD_DIM))


model.fit_generator(generator(X, Ymatrix), steps_per_epoch=len(Ymatrix), epochs=1)
