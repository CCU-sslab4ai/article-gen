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

## LICENSE

Copyright (C) 2018  Yu-Jhe, Gao

This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
This is free software, and you are welcome to redistribute it under certain conditions; type `show c' for details.

The hypothetical commands `show w' and `show c' should show the appropriate
parts of the General Public License.  Of course, your program's commands
might be different; for a GUI interface, you would use an "about box".

You should also get your employer (if you work as a programmer) or school,
if any, to sign a "copyright disclaimer" for the program, if necessary.
For more information on this, and how to apply and follow the GNU GPL, see
<http://www.gnu.org/licenses/>.

The GNU General Public License does not permit incorporating your program
into proprietary programs.  If your program is a subroutine library, you
may consider it more useful to permit linking proprietary applications with
the library.  If this is what you want to do, use the GNU Lesser General
Public License instead of this License.  But first, please read
<http://www.gnu.org/philosophy/why-not-lgpl.html>.