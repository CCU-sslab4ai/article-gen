# -*- coding=UTF-8 -*-
def encode(s):
    try:
        return chr(s)
    except:
        return 'Encode Error'


def decode(s):
    try:
        return ord(s)
    except:
        return 'Decode Error'


def str2arr(s):
    arr = []
    for i in range(0, len(s)):
        arr.append(int(s[i]))

    return arr


def arr2str(arr):
    strArr = map(str, arr)
    return ''.join(strArr)


def int2bin(i):
    return bin(i)[2:].zfill(16)


def bin2int(b):
    return int(b, 2)
