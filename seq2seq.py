from keras.layers import Input, LSTM
from keras.models import Model

def model1():
    encoderInputs = Input(shape=(None, 16))
    encoderHidden = LSTM(1, return_state=True)
    encoderOutputs, h, c = encoderHidden(encoderInputs)
    encoderStates = [h, c]

    decoderInputs = Input(shape=(None, 16))
    decoderHidden = LSTM(1, return_sequences=True, return_state=True)
    decoderOutputs, _, _ = decoderHidden(
        decoderInputs, initial_state=encoderStates)

    return Model([encoderInputs, decoderInputs], decoderOutputs)
