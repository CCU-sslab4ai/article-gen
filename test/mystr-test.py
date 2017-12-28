import mystr
import s2s

input = '台灣 2040 年禁售燃油車算晚？看看其他國家時間表就知道！'
X = []

for i in range(0, len(input)):
    unicode10 = mystr.decode(input[i])
    unicode2 = mystr.int2bin(unicode10)
    oneHot = mystr.str2arr(unicode2)
    X.append(oneHot)

print(X)

Y = ''
for i in range(0, len(input)):
    unicode2 = mystr.arr2str(X[i])
    unicode10 = mystr.bin2int(unicode2)
    Y += mystr.encode(unicode10)

print(Y)