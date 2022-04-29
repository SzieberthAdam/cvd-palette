# 3rd party libraries
from PIL import Image
import numpy as np

# standard libraries
from decimal import Decimal as D, getcontext
import pathlib
import sys

getcontext().prec = 30


def sRGBtoLin(v):
    if not isinstance(v, D):
        v = D(str(v))
    if v <= D("0.04045"):
        return v / D("12.92")
    else:
        return ((v + D("0.055")) / D("1.055"))**D("2.4")

def LintosRGB(v):
    if not isinstance(v, D):
        v = D(str(v))
    if v <= D("0.0031308"):
        v_ = v * D("12.92")
    else:
        v_ = D("1.055") * v**(D("1")/D("2.4")) - D("0.055")
    return max(D(0), min(D(255), round(v_ * D(255))))

def RGBtoY(RGB):
    sRGB = tuple(D(x) for x in RGB)
    vRGB = tuple(x/255 for x in sRGB)
    lRGB = tuple(sRGBtoLin(x) for x in vRGB)
    Y = D("0.2126") * lRGB[0] + D("0.7152") * lRGB[0] + D("0.0722") * lRGB[0]
    return Y

def YtoLstar(Y):
    if Y <= D(216)/D(24389):  # 0.008856
        return Y * D(24389)/D(27)  # 903.3
    else:
        return Y**(D(1)/D(3)) * D(116) - D(16)

def gen_identityRGB():
    for b in range(256):
        for g in range(256):
            for r in range(256):
                yield D(r), D(g), D(b)

def getYarr():

    for i, RGB in enumerate(gen_identityRGB()):
        arr[i] = RGBtoY(RGB)
    return arr

def getlinYarr(Yarr=None):
    Yarr = Yarr or getYarr()
    return np.vectorize(LintosRGB)(Yarr)


if __name__ == "__main__":
    Ys = [None] * 4096 * 4096
    linYarr = np.empty(4096*4096, dtype="uint8")
    for i, RGB in enumerate(gen_identityRGB()):
        if RGB[0] == 0 and RGB[1] == 0:
            print(RGB[2])
        Ys[i] = Y = RGBtoY(RGB)
        linYarr[i] = linY = LintosRGB(Y)
