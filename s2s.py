from keras.models import Sequential
from keras.layers import LSTM, Embedding
from keras.layers.core import Reshape
from keras import layers


def model1(INPUT_MAX_LEN, WORD_DIM, OUTPUT_MAX_LEN, HIDDEN_DIM, DEPTH=1):
    model = Sequential()
    model.add(Embedding(WORD_DIM, HIDDEN_DIM, input_length=INPUT_MAX_LEN, mask_zero=True))
    model.add(LSTM(HIDDEN_DIM))
    model.add(layers.RepeatVector(OUTPUT_MAX_LEN))
    for _ in range(DEPTH):
        model.add(LSTM(HIDDEN_DIM, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(WORD_DIM)))
    model.add(layers.Activation('softmax'))
    return model
