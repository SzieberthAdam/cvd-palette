# standard libraries
import bisect
import collections
from datetime import datetime
import itertools
import json
import math
import pathlib
import sys

_thisdir = pathlib.Path(__file__).parent.resolve()
_importdir = _thisdir.parent.resolve()
sys.path.insert(0, str(_importdir))

# 3rd party libraries (module file available)
import clut
import de2000
import pal
import rgbpyramid

sys.path.remove(str(_importdir))

# 3rd party libraries
import colour
from PIL import Image
import numpy as np

def delta_e(lab1, lab2):
    lab1_ = np.array(lab1)
    lab2_ = np.array(lab2)
    delta_E = colour.difference.delta_E_CIE2000(lab1_, lab2_)
    return delta_E

def rgbarr_to_labarr(arr):
    arr = colour.sRGB_to_XYZ(arr/255)
    arr = colour.XYZ_to_Lab(arr)
    return arr

def rgb_to_lab(rgb):
    arr = rgbarr_to_labarr(np.array(list(rgb), dtype=np.uint8).reshape((1,1,3)))
    return tuple(arr)

def rgbstr(rgb):
    r, g, b = rgb
    return f'#{r:0>2x}{g:0>2x}{b:0>2x}'

def palstr(palarr):
    return ", ".join(rgbstr(rgb) for rgb in palarr)


def idxtcnt(pools, nums=None):
    nums = nums or [1] * len(pools)
    v = 1
    for i, pool in enumerate(pools):
        v *= math.comb(len(pool),nums[i])
    return v

def idxtgen(pools, nums=None, i=0):
    nums = nums or [1] * len(pools)
    for a in itertools.combinations(tuple(range(len(pools[0]))), nums[0]):
        t = tuple((i, v) for v in a)
        if len(pools) == 1:
            yield t
        else:
            for b in idxtgen(pools[1:], nums=nums[1:], i=i+1):
                yield t + b

class keydefaultdict(collections.defaultdict):
    def __missing__(self, key):
        ret = self[key] = self.default_factory(key)
        return ret


if __name__ == "__main__":

    json_dump_kwargs = {'ensure_ascii': False, 'indent': '\t', 'sort_keys': True}

    if len(sys.argv) < 2 or 3 < len(sys.argv) or not sys.argv[1].isdecimal():
        print("Integer CVD number is expected as first command line argument!")
        sys.exit(1)
    else:
        cvd_n = int(sys.argv[1])
    max_dE = None

    root = pathlib.Path(__file__).parent.resolve()
    cvdpaldir = root / f'cvd{cvd_n}'

    haldclutdir = root.parent / "haldclut"

    colviss = {
        "normal":       clut.CLUT(str(haldclutdir / "identity" / "identity.png")),
        "deuteranopia": clut.CLUT(str(haldclutdir / "cvd" / "deuta.machado2010.png")),
        "protanopia":   clut.CLUT(str(haldclutdir / "cvd" / "prota.machado2010.png")),
        "tritanopia":   clut.CLUT(str(haldclutdir / "cvd" / "trita.machado2010.png")),
    }

    graypalroot = root.parent / "grayscale-palette"
    graypalp = graypalroot / f'grayscale{cvd_n:0>2}.pal'
    with graypalp.open() as f:
        graypal = tuple(rgb[0] for rgb in pal.load(f))

    midgrays = tuple(graypal[1:-1])
    isoluminant_fnames = {v: f'{v:x}.png' for v in midgrays}

    level = 2

    cntcap = 55555
    level = 256
    colorpools_rgbs = {}
    while True:
        leveldir = root.parent / "isoluminant" / f'level{level:0>3}'

        colorpools_rgbs[level] = colorpools_rgb = []
        for v in graypal:
            if v in {0, 255}:
                colorpools_rgb.append(np.array(((v, v, v),), dtype="uint8"))
            else:
                img = Image.open(str(leveldir / isoluminant_fnames[v]))
                arr = np.array(img).reshape((-1, 3))
                colorpools_rgb.append(arr)
        cnt = idxtcnt(colorpools_rgb)
        if level==2 or cnt <= cntcap:
            break
        else:
            level = level // 2

    high_n = 100

    while True:
        print(f'=== LEVEL: {level}')
        highs = collections.defaultdict(set)
        high_dE = []
        high_pal = []

        cv_colorpools_rgb = {}
        cv_colorpools_lab = {}
        for cv, cvhaldclut in colviss.items():
            cv_colorpools_rgb[cv] = [cvhaldclut.clut[a[:,0], a[:,1], a[:,2]] for a in colorpools_rgb]
            cv_colorpools_lab[cv] = [de2000.get_lab_arr(rgb_arr) for rgb_arr in cv_colorpools_rgb[cv]]

        cnt = idxtcnt(colorpools_rgb)
        for c, idxt in enumerate(idxtgen(colorpools_rgb), 1):
            if c==1 or not c % 100000:
                print(f'at {c} of {cnt}')
            #print(c)
            rgb_palette = tuple(tuple(colorpools_rgb[i][j]) for i, j in idxt)
            idxt_dE = (999999999,)
            for cv, lab_pool in cv_colorpools_lab.items():
                lab_arr = np.array([lab_pool[i][j] for i, j in idxt], dtype="float64")
                dE = de2000.get_pal_delta_e_from_lab_arr(lab_arr)
                if dE < idxt_dE:
                    idxt_dE = dE
            highs[idxt_dE].add(rgb_palette)

        if level == 256:
            break
        else:
            old_level = level
            level *= 2
            new_cnt = 0
            colorpools_rgb_set = [set() for _ in range(cvd_n)]
            dEsorted = sorted(highs, reverse=True)
            print(f'dE = {dEsorted[0][0]}')
            ref_cnt = 0
            for dE in dEsorted:
                pal_rgbs = highs[dE]
                for pal_rgb in pal_rgbs:
                    for i, rgb in enumerate(pal_rgb):
                        if rgb in {(0, 0, 0), (255, 255, 255)}:
                            new_set = {rgb}
                        else:
                            ref_rgb = rgbpyramid.get_ref_rgb(rgb, old_level)
                            new_set = set(
                                tuple(rgb1) for rgb1 in colorpools_rgbs[level][i]
                                if rgbpyramid.get_ref_rgb(rgb1, old_level) == ref_rgb
                            )
                        colorpools_rgb_set[i] |= new_set
                new_cnt = idxtcnt(colorpools_rgb_set)
                assert new_cnt
                if 2*ref_cnt <= new_cnt:
                    print(f'at {new_cnt} / {cntcap} for level {level}')
                    ref_cnt = new_cnt
                if cntcap < new_cnt:
                    break
            colorpools_rgb = [np.array(tuple(s),dtype="uint8") for s in colorpools_rgb_set]

    high_dE = []
    high_pal = []
    for dE in dEsorted:
        for rgb_palette in highs[dE]:
            high_dE.append(dE)
            high_pal.append(rgb_palette)
        if high_n <= len(high_pal):
            break

    valstrparts = []
    for i, t in enumerate(high_dE):
        _hstr = f'{i}: '
        _vstr = "   ".join([f'{x:0<18}'[:18] for x in t])
        valstrparts.append(f'{_hstr:<6}{_vstr}')
    cvdpaldir.mkdir(parents=True, exist_ok=True)
    with (cvdpaldir / f'cvd{cvd_n}.top{high_n}.de2000.txt').open("w") as f:
        f.write("\n".join(valstrparts) + "\n")

    high_pal_arr = np.array(tuple(high_pal), dtype="uint8")
    high_pal_img = Image.fromarray(high_pal_arr, 'RGB')
    high_pal_img.save(cvdpaldir / f'cvd{cvd_n}.top{high_n}.png')
