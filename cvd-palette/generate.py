# standard libraries
import collections
import datetime
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

json_dump_kwargs = {'ensure_ascii': False, 'indent': '\t', 'sort_keys': True}

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

    np.seterr(all='raise')

    cvd_n = int(sys.argv[1])
    start_level = int(sys.argv[2])
    trdistrate = float(sys.argv[3])
    trtoppop = int(sys.argv[4])
    trratepop = int(sys.argv[5])
    if trratepop:
        trrate = float(sys.argv[6])
        print(f'Making CVD{cvd_n} palette from start level {start_level} with transfer params: {trdistrate}, {trtoppop}, {trratepop}, {trrate} ...')
    else:
        print(f'Making CVD{cvd_n} palette from start level {start_level} with transfer params: {trdistrate}, {trtoppop}, {trratepop} ...')

    root = pathlib.Path(__file__).parent.resolve()
    cvdpaldir = root / f'cvd{cvd_n}'
    cvdpaldir.mkdir(parents=True, exist_ok=True)

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
    
    colorpools_rgbs = {}
    gen_level = start_level
    while gen_level <= 256:
        gen_leveldir = root.parent / "isoluminant" / f'level{gen_level:0>3}'

        colorpools_rgbs[gen_level] = colorpools_rgb = []
        for v in graypal:
            if v in {0, 255}:
                colorpools_rgb.append(np.array(((v, v, v),), dtype="uint8"))
            else:
                img = Image.open(str(gen_leveldir / isoluminant_fnames[v]))
                arr = np.array(img).reshape((-1, 3))
                colorpools_rgb.append(arr)
        cnt = idxtcnt(colorpools_rgb)
        gen_level *= 2

    level = start_level
    colorpools_rgb = colorpools_rgbs[level]
    
    while True:
        print(f'=== LEVEL: {level}')
        
        highs = collections.defaultdict(set)
        highs_pop = 0

        cv_colorpools_rgb = {}
        cv_colorpools_lab = {}
        for cv, cvhaldclut in colviss.items():
            cv_colorpools_rgb[cv] = [cvhaldclut.clut[a[:,0], a[:,1], a[:,2]] for a in colorpools_rgb]
            cv_colorpools_lab[cv] = [de2000.get_lab_arr(rgb_arr) for rgb_arr in cv_colorpools_rgb[cv]]

        cnt = idxtcnt(colorpools_rgb)
        for c, idxt in enumerate(idxtgen(colorpools_rgb), 1):
            if c==1 or not c % 100000:
                print(f'{datetime.datetime.now()}: at {c} of {cnt}')
            #print(c)
            rgb_palette = tuple(tuple(colorpools_rgb[i][j]) for i, j in idxt)
            idxt_dE = (999999999,)
            for cv, lab_pool in cv_colorpools_lab.items():
                lab_arr = np.array([lab_pool[i][j] for i, j in idxt], dtype="float64")
                dE = de2000.get_pal_delta_e_from_lab_arr(lab_arr)
                if dE < idxt_dE:
                    idxt_dE = dE
            highs[idxt_dE].add(rgb_palette)
            highs_pop += 1

            if (highs_pop % 250000) == 0:  # save some memory
                print("combinations at 250000; cutting back...")
                dEsorted = sorted(highs)
                for dE in dEsorted:
                    c = len(highs[dE])
                    del highs[dE]
                    highs_pop -= c
                    if highs_pop <= 100000:
                        break
                print(f'{highs_pop} combinations were kept.')

        dEsorted = sorted(highs, reverse=True)
        print(f'dE = {dEsorted[0][0]}')
        
        serobj = []
        for dE in dEsorted:
            serobj.append(["dE", dE])
            for pal_rgb in sorted(highs[dE]):
                serobj.append(["pal", pal_rgb])
        with (cvdpaldir / f'cvd{cvd_n}.highs.level{level}.json').open("w") as f:
            json.dump(serobj, f)
        del serobj
                                           
        topN_dE = []
        NtopN_dE = len(dEsorted[:trtoppop])
        for dE in dEsorted[:trtoppop]:
            for pal_rgb in sorted(highs[dE]):
                topN_dE.append((dE, pal_rgb))
        
        max_dE = dEsorted[0][0]
        trans_dE = []
        if trratepop:
            for dE in dEsorted[trtoppop:]:
                if trrate * dE[0] < max_dE:
                    break
                for pal_rgb in sorted(highs[dE]):
                    trans_dE.append((dE, pal_rgb))
            Ntrans_dE = len(set(t[0] for t in trans_dE))
            trans_dE = trans_dE[::max(1, len(trans_dE)//trratepop)]
            print(f'transfer dE = {trans_dE[-1][0]}')
        else:
            Ntrans_dE = 0

        with (cvdpaldir / f'cvd{cvd_n}.trans{level}.txt').open("a") as f:
            prev_dE = None
            f.write(f'=== PALETTES OF TOP {trtoppop} dE2000 TRANSFERRED; dE COUNT {NtopN_dE}; PALETTE COUNT = {len(topN_dE)}  ===\n')
            for (dE, pal_rgb) in topN_dE:
                if prev_dE is None or set(dE) - set(prev_dE):
                    f.write(", ".join([f'{v}' for v in dE]) + "\n")
                    prev_dE = dE
                f.write(de2000.palstr(pal_rgb) + "\n")
            if trratepop:
                f.write(f'=== PALETTES OF dE RATE {trratepop} TRANSFERRED; dE COUNT {Ntrans_dE}; PALETTE COUNT = {len(trans_dE)}  ===\n')
                for dE in trans_dE:
                    if prev_dE is None or set(dE) - set(prev_dE):
                        f.write(", ".join([f'{v}' for v in dE]) + "\n")
                        prev_dE = dE
                    f.write(de2000.palstr(pal_rgb) + "\n")
                         
        if level == 256:
            break

        old_level = level
        level *= 2
        distancecap = trdistrate * (256 / old_level)
        new_cnt = 0
        colorpools_rgb_set = [set() for _ in range(cvd_n)]
        colorpools_rgb_set[0] = {(0, 0, 0)}
        colorpools_rgb_set[-1] = {(255, 255, 255)}

        for (dE, pal_rgb) in topN_dE:
            for i, rgb in enumerate(pal_rgb[1:-1], 1):
                if rgb not in colorpools_rgb_set[i]:
                    new_set = {rgb}
                    for rgb1 in colorpools_rgbs[level][i]:
                        distance = rgbpyramid.get_distance_fast(rgb, rgb1)
                        if distance <= distancecap:
                            new_set.add(tuple(rgb1))  # rgb1 is np.ndarray of np.uint8
                        colorpools_rgb_set[i] |= new_set
        new_cnt_top = idxtcnt(colorpools_rgb_set)
        print(f'{new_cnt_top} palettes transferred to level {level} by top {len(topN_dE)} level {old_level} palettes')
        if trratepop:
            for (dE, pal_rgb) in trans_dE:
                for i, rgb in enumerate(pal_rgb[1:-1], 1):
                    if rgb not in colorpools_rgb_set[i]:
                        new_set = {rgb}
                        for rgb1 in colorpools_rgbs[level][i]:
                            distance = rgbpyramid.get_distance_fast(rgb, rgb1)
                            if distance <= distancecap:
                                new_set.add(tuple(rgb1))  # rgb1 is np.ndarray of np.uint8
                            colorpools_rgb_set[i] |= new_set
            new_cnt = idxtcnt(colorpools_rgb_set)
            print(f'{new_cnt-new_cnt_top} palettes transferred to level {level} by {len(trans_dE)} transfer palettes of level {old_level}')
                            
        colorpools_rgb = [np.array(tuple(s),dtype="uint8") for s in colorpools_rgb_set]        

    with (cvdpaldir / f'cvd{cvd_n}.de2000.txt').open("w") as f:
        f.write("   ".join([f'{x:0<18}'[:18] for x in topN_dE[0][0]]) + "\n")
        
    high_pal_arr = np.array(tuple(rgb_pal for (dE, rgb_pal) in topN_dE if dE==dEsorted[0]), dtype="uint8")
    high_pal_img = Image.fromarray(high_pal_arr, 'RGB')
    high_pal_img.save(cvdpaldir / f'cvd{cvd_n}.top.png')
