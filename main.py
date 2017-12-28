import mystr
import s2s
from keras.utils import plot_model

input = '台灣 2040 年禁售燃油車算晚？看看其他國家時間表就知道！'
X = []

for i in range(0, len(input)):
    unicode10 = mystr.decode(input[i])
    unicode2 = mystr.int2bin(unicode10)
    oneHot = mystr.str2arr(unicode2)
    X.append(oneHot)
    # print(input[i] + ' : ' + str(oneHot))

print(X)

model = s2s.model2()
model.compile(loss='mse', optimizer='rmsprop')
