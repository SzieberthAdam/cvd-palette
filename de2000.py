import collections
import itertools
import sys

# 3rd party libraries
import colour
import numpy as np
from PIL import Image

def delta_e_from_lab(lab1, lab2):
    delta_E = colour.difference.delta_E_CIE2000(lab1, lab2)
    return delta_E

def get_lab_arr(rgb_arr):
    xyz_arr = colour.sRGB_to_XYZ(rgb_arr/255)
    lab_arr = colour.XYZ_to_Lab(xyz_arr)
    return lab_arr.reshape(rgb_arr.shape)

def get_pal_delta_e_pairs(rgb_arr):
    L = []
    for comb_ in itertools.combinations(rgb_arr, 2):
        comb = tuple(tuple(rgb) for rgb in comb_)
        comb_arr = np.array(comb, dtype="uint8")
        dEval = get_pal_delta_e(comb_arr)[0]
        L.append((dEval, comb))
    return tuple(sorted(L))

def get_pal_delta_e_pairs_report(rgb_arr):
    S = []
    for dEval, comb in get_pal_delta_e_pairs(rgb_arr):
        S.append(f'{palstr(comb)} -> {dEval}')
    return "\n".join(S) + "\n"

def get_pal_delta_e_from_lab_arr(lab_arr):
    combs = list(itertools.combinations(lab_arr, 2))
    lab1_arr, lab2_arr = np.array(list(zip(*combs)))
    delta_e_arr = delta_e_from_lab(lab1_arr, lab2_arr)
    return tuple(sorted(delta_e_arr))

def get_pal_delta_e(rgb_arr):
    lab_arr = get_lab_arr(rgb_arr)
    return get_pal_delta_e_from_lab_arr(lab_arr)

def palstr(palarr):
    return " ".join(rgbstr(rgb) for rgb in palarr)

def palstr2pal(s):
    return tuple(str2rgb(s_) for s_ in s.split())

def rgbstr(rgb):
    r, g, b = rgb
    return f'#{r:0>2X}{g:0>2X}{b:0>2X}'


def str2rgb(s):
    s = s.strip(" #,\n")
    t = tuple(int("0x"+"".join(t), 16) for t in zip(s[::2], s[1::2]))
    return t


def main(*argv):
    try:
        rgb_arr = np.array([str2rgb(s) for s in sys.argv[1:]], dtype="uint8")
    except (ValueError, AssertionError):
        print(f'ERROR! At least two HTML format colors were expected as arguments.')
        return 1

    print(get_pal_delta_e_pairs_report(rgb_arr), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv))
