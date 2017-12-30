from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.core import Dense, RepeatVector


def model1(input_dim, hidden_dim, max_in_seq_len=50, max_out_seq_len=1000):
    model = Sequential()
    model.add(Bidirectional(GRU(input_dim=input_dim, output_dim=hidden_dim,
                                return_sequences=True), input_shape=(max_in_seq_len, input_dim)))
    model.add(Dense(hidden_dim, activation='softmax'))
    model.add(RepeatVector(max_out_seq_len))
    model.add(Bidirectional(GRU(hidden_dim, return_sequences=True)))
    model.add(TimeDistributed(Dense(output_dim=input_dim, activation='softmax')))

    return model
