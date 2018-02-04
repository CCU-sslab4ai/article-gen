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
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import losses
from keras import metrics
from keras import backend as K

from time import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

MAX_INPUT = 98
MIN_OUTPUT = 304
MAX_OUTPUT = 872
WORD_DIM = 21574
TRAIN_SIZE = 408
VAL_SIZE = int(TRAIN_SIZE * 0.25)

csvArr = myfile.readcsv('./datasets/legend-of-the-white-snake2.csv')

inputArr = []
outputArr = []
X = np.array([]).reshape(-1, MAX_INPUT)

Ymatrix = []

print('讀取csv')
for i in trange(0, len(csvArr)):
    inputArr.append(csvArr[i][0])
    outputArr.append(csvArr[i][1])

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

Xtrain = X[:TRAIN_SIZE]
Ytrain = Ymatrix[:TRAIN_SIZE]
Xval = X[TRAIN_SIZE:(TRAIN_SIZE + VAL_SIZE)]
Yval = Ymatrix[TRAIN_SIZE:(TRAIN_SIZE + VAL_SIZE)]
# Xtest = X[TRAIN_SIZE:]
# Ytest = Ymatrix[TRAIN_SIZE:]


def customLoss(Y_true, Y_pred):
    return losses.categorical_crossentropy(Y_true[:, :MIN_OUTPUT, :], Y_pred[:, :MIN_OUTPUT, :])


def customAcc(Y_true, Y_pred):
    return K.sum(K.mean(K.equal(K.argmax(Y_pred[:, :MIN_OUTPUT, :], axis=2), K.argmax(Y_true[:, :MIN_OUTPUT, :], axis=2)), axis=1))

model = s2s.model1(MAX_INPUT, WORD_DIM, MAX_OUTPUT, 64)
model.summary()
model.compile(loss=customLoss,
              optimizer='adam', metrics=['acc'])


def generator(Xnp, YSparse):
    while 1:
        for i in range(0, len(YSparse)):
            yield Xnp[i].reshape(-1, MAX_INPUT), YSparse[i].toarray().reshape((-1, MAX_OUTPUT, WORD_DIM))


checkpointer = ModelCheckpoint(
    filepath='./model/weights-customLoss.hdf5', verbose=1, save_best_only=True)

tensorboard = TensorBoard(log_dir="./model/logs/{}".format(time()))

history = model.fit_generator(generator=generator(Xtrain, Ytrain), steps_per_epoch=TRAIN_SIZE, validation_data=generator(
    Xval, Yval), validation_steps=VAL_SIZE, epochs=100)


# cost = model.evaluate_generator(
#     generator(Xtest, Ytest), steps=len(Ytest))
# print('test cost' + str(cost))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./model/acc.png')
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./model/loss.png')
