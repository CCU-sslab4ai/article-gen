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


def int2bin(i):
    return bin(i)[2:].zfill(16)