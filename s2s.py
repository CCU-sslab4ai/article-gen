from keras.models import Sequential
from keras.layers import LSTM, Embedding
from keras import layers


def model1(INPUT_MAX_LEN, WORD_DIM, OUTPUT_MAX_LEN, HIDDEN_DIM, DEPTH=1):
    model = Sequential()
    model.add(Embedding(WORD_DIM, HIDDEN_DIM, input_length=INPUT_MAX_LEN))
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_DIM.
    # Note: In a situation where your input sequences have a variable length,
    # use input_shape=(None, num_feature).
    model.add(LSTM(HIDDEN_DIM))
    # As the decoder RNN's input, repeatedly provide with the last hidden state of
    # RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
    # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
    model.add(layers.RepeatVector(OUTPUT_MAX_LEN))
    # The decoder RNN could be multiple layers stacked or a single layer.
    for _ in range(DEPTH):
        # By setting return_sequences to True, return not only the last output but
        # all the outputs so far in the form of (num_samples, timesteps,
        # output_dim). This is necessary as TimeDistributed in the below expects
        # the first dimension to be the timesteps.
        model.add(LSTM(HIDDEN_DIM, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(WORD_DIM)))
    model.add(layers.Activation('softmax'))
    return model
