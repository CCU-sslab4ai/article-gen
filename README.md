# Article Gen

## Quick Start

```
$ mkdir model
$ python train.py
```

## Seq2Seq主體模型

在s2s.py內，目前確定穩定版本如下

```python
model = Sequential()
model.add(Embedding(WORD_DIM, HIDDEN_DIM, input_length=INPUT_MAX_LEN))
model.add(LSTM(HIDDEN_DIM))
model.add(layers.RepeatVector(OUTPUT_MAX_LEN))
for _ in range(DEPTH):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(WORD_DIM)))
model.add(layers.Activation('softmax'))
```