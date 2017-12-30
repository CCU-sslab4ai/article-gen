from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, RepeatVector

def model1(input_dim, max_out_seq_len, hidden_dim):
    model = Sequential()
    model.add(GRU(input_dim=input_dim, output_dim=hidden_dim, return_sequences=False))
    model.add(Dense(hidden_dim, activation='softmax'))
    model.add(RepeatVector(max_out_seq_len))
    model.add(GRU(hidden_dim, return_sequences=True))
    model.add(TimeDistributed(Dense(output_dim=input_dim, activation='softmax')))

    return model