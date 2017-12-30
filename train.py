import s2s
import mystr
import myfile
import numpy as np

MAX_INPUT=50
MAX_OUTPUT=1000

csvArr=myfile.readcsv('./datasets/news20171225.csv')

inputArr=[]
outputArr=[]
for i in range(0, len(csvArr)):
    inputArr.append(csvArr[i][0])
    outputArr.append(csvArr[i][2])

X = []
Y = []

for i in range(0, len(inputArr)):
    input = inputArr[i]
    output = outputArr[i]
    tempIn = []
    tempOut = []

    for j in range(0, MAX_INPUT):
        try:
            unicode10 = mystr.decode(input[j])
            unicode2 = mystr.int2bin(unicode10)
            oneHot = mystr.str2arr(unicode2)
            tempIn.append(oneHot)
        except IndexError:
            tempIn.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        

    for j in range(0, MAX_OUTPUT):
        try:
            unicode10 = mystr.decode(output[j])
            unicode2 = mystr.int2bin(unicode10)
            oneHot = mystr.str2arr(unicode2)
            tempOut.append(oneHot)
        except IndexError:
            tempOut.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        

    X.append(tempIn)
    Y.append(tempOut)

Xnp = np.array(X)
Ynp = np.array(Y)

model = s2s.model1(input_dim=16, hidden_dim=16)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
model.fit(Xnp, Ynp, epochs=50)
