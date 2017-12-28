from keras.layers import Input, LSTM
from keras.models import Model
import seq2seq
from seq2seq.models import SimpleSeq2Seq


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


def model2():
    model = SimpleSeq2Seq(input_dim=400, hidden_dim=10, output_length=20000, output_dim=20000, depth=3)
    return model
    