import s2s
import mystr
import numpy as np

inputArr = ['保密到家！ 內政部、婦聯會 明簽行政契約備忘錄']
outputArr = ['〔記者鍾麗華／台北報導〕內政部、黨產會與婦聯會明天將正式簽署行政契約備忘錄，並召開記者會並對外公佈簽署內容，不過，由於還有相關的細節，尚待各自帶回協調，因此內政部今天對簽署內容三緘其口，並不願透露。而婦聯會表示，這次備忘錄內容與上次相差並不大。\n\n婦聯會今天下午舉行常委會，與會人員全數同意與內政部簽訂行政契約備忘錄。常委會結束後，約晚間6點多，婦聯會主委雷倩帶著2名律師到內政部長葉俊榮辦公室說明。據了解，黨產會與民政司長林清淇並未與會。而葉俊榮與婦聯會談了一個多小時，結束後雷倩刻意從地下室離開，避開媒體。\n\n婦聯會律師張菀萱表示，今晚主要是向葉俊榮報告下午決議通過的內容，大致方向已經談妥，也有初步共識，主要是在上次行政契約的基礎之上，不過，詳細文本還要帶回進一步磋商，明天會簽署備忘錄，也會對外公佈備忘錄內容。']
X = []
Y = []

for i in range(0, len(inputArr)):
    input = inputArr[i]
    output = outputArr[i]
    tempIn = []
    tempOut = []

    for j in range(0, len(input)):
        unicode10 = mystr.decode(input[j])
        unicode2 = mystr.int2bin(unicode10)
        oneHot = mystr.str2arr(unicode2)
        tempIn.append(oneHot)

    for j in range(0, len(output)):
        unicode10 = mystr.decode(output[j])
        unicode2 = mystr.int2bin(unicode10)
        oneHot = mystr.str2arr(unicode2)
        tempOut.append(oneHot)

    X.append(tempIn)
    Y.append(tempOut)

Xnp = np.array(X, ndmin=3)
Ynp = np.array(Y, ndmin=3)

model = s2s.model2()
model.compile(loss='mse', optimizer='sgd')
model.fit(Xnp, Ynp, epochs=1)
