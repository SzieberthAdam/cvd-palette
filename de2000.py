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
    return lab_arr

def get_pal_delta_e(rgb_arr):
    lab_arr = get_lab_arr(rgb_arr)
    combs = list(itertools.combinations(lab_arr, 2))
    lab1_arr, lab2_arr = np.array(list(zip(*combs)))
    delta_e_arr = delta_e_from_lab(lab1_arr, lab2_arr)
    return delta_e_arr.min()

def palstr(palarr):
    return ", ".join(rgbstr(rgb) for rgb in palarr)

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

    dEvals = collections.defaultdict(set)
    for n, comb in enumerate(itertools.combinations(rgb_arr, 2), 1):
        comb = tuple(tuple(rgb) for rgb in comb)
        comb_arr = np.array(comb, dtype="uint8")
        dEval = get_pal_delta_e(comb_arr)
        dEvals[dEval].add(comb)

    pal_dEval = min(dEvals)

    for i, dEval in enumerate(sorted(dEvals)):
        print(f'[ dE2000 = {dEval} {"(palette Delta-E) " if i==0 and 2 < len(rgb_arr) else ""}]')
        for comb in sorted(dEvals[dEval]):
            print(f'    {palstr(comb)}')

    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv))
