# standard libraries
import collections
import datetime
import itertools
import json
import math
import pathlib
import random
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

LEVELS = 2, 4, 8, 16, 32, 64, 128, 256

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

def write_highs(highs, filepath):
    dEsorted = sorted(highs, reverse=True)
    parts = []
    for dE in dEsorted:
        parts.append(", ".join([f'{v}' for v in dE]))
        for pal_rgb in sorted(highs[dE]):
            parts.append(de2000.palstr(pal_rgb))
    with filepath.open("w") as f:
        f.write("\n".join(parts))


def read_highs(filepath):
    highs = collections.defaultdict(set)
    with filepath.open("r") as f:
        for line in f.readlines():
            if not line.startswith("#"):
                dE = tuple(float(s.strip()) for s in line.split(","))
            else:
                pal_rgb = de2000.palstr2pal(line)
                highs[dE].add(pal_rgb)
    return highs


class keydefaultdict(collections.defaultdict):
    def __missing__(self, key):
        ret = self[key] = self.default_factory(key)
        return ret


if __name__ == "__main__":

    np.seterr(all='raise')

    root = pathlib.Path(__file__).parent.resolve()

    haldclutdir = root.parent / "haldclut"

    colviss = {
        "normal":       clut.CLUT(str(haldclutdir / "identity" / "identity.png")),
        "deuteranopia": clut.CLUT(str(haldclutdir / "cvd" / "deuta.machado2010.png")),
        "protanopia":   clut.CLUT(str(haldclutdir / "cvd" / "prota.machado2010.png")),
        "tritanopia":   clut.CLUT(str(haldclutdir / "cvd" / "trita.machado2010.png")),
    }


    highsfp = None
    errormsg = "ERROR! palette size or HIGHS file was expected as first argument."
    try:
        cvd_n = int(sys.argv[1])
    except IndexError:
        print(errormsg)
        sys.exit(1)
    except ValueError:
        highsfp = pathlib.Path(sys.argv[1])
        if not highsfp.is_file():
            highsfp = cvdpaldir / sys.argv[1]
        if not highsfp.is_file():
            print(errormsg)
            sys.exit(2)
    levelat_c = -1
    if highsfp:
        parts = tuple(s.lower() for s in highsfp.stem.split("."))
        assert parts[0].startswith("cvd")
        assert parts[1] == "highs"
        cvd_n = int(parts[0][3:])
        start_level = int(parts[2])
        level = int(parts[3])
        highs = read_highs(highsfp)
        highs_pop = 0
        for s_ in highs.values():
            highs_pop += len(s_)
        if 4 < len(parts):
            assert parts[4] == "at"
            levelat_c = int(parts[4])
        else:
            levelat_c = highs_pop


    print(f'Making CVD{cvd_n} palette.')
    cvdpaldir = root / f'cvd{cvd_n}'

    graypalroot = root.parent / "grayscale-palette"
    graypalp = graypalroot / f'grayscale{cvd_n:0>2}.pal'
    with graypalp.open() as f:
        graypal = tuple(rgb[0] for rgb in pal.load(f))

    graypalstrs = tuple(f'{v:0>2x}' for v in graypal)

    print(f'Gray levels in hexadecimal: {", ".join(graypalstrs)}')

    colorpools_rgbs = {}
    for level_ in LEVELS:
        leveldir = root.parent / "isoluminant" / f'level{level_:0>3}'
        colorpools_rgbs[level_] = colorpools_rgb = []
        for v in graypal:
            if v in {0, 255}:
                colorpools_rgb.append(np.array(((v, v, v),), dtype="uint8"))
            else:
                img = Image.open(str(leveldir / f'{v:0>2x}.png'))
                arr = np.array(img).reshape((-1, 3))
                colorpools_rgb.append(arr)
        if not highsfp:
            cnt = idxtcnt(colorpools_rgb)
            if cnt < 10000000:
                print(f'Palette count for start level {level_} = {cnt}')
            else:
                print(f'Palette count for start level {level_} = {cnt:.4E}')

    if not highsfp:
        answer = ""
        while answer not in {str(x) for x in LEVELS}:
            print("Please give me the desired start palette count")
            print("or type Q or E for exit.")
            answer = input("::: ").lower()
            if answer in {"q", "e"}:
                sys.exit(100)

        level = start_level = int(answer)

    colorpools_rgb = colorpools_rgbs[level]

    total_time = datetime.timedelta(0)
    total_cnt = 0

    if not highsfp:

        # exceptional first pass

        print(f'=== START LEVEL: {level}')

        if not highsfp:
            highs = collections.defaultdict(set)
            highs_pop = 0

        cv_colorpools_rgb = {}
        cv_colorpools_lab = {}
        for cv, cvhaldclut in colviss.items():
            cv_colorpools_rgb[cv] = [cvhaldclut.clut[a[:,0], a[:,1], a[:,2]] for a in colorpools_rgb]
            cv_colorpools_lab[cv] = [de2000.get_lab_arr(rgb_arr) for rgb_arr in cv_colorpools_rgb[cv]]

        start_tcnt = tcnt = idxtcnt(colorpools_rgb)

        start_level_start_time = level_start_time = datetime.datetime.now()
        print(f'start time: {level_start_time:%Y-%m-%d %H:%M:%S}')

        prevprinttime = level_start_time
        prevprintc = 0
        nextprintc = 1

        prevatfp = None

        try:
            for c, idxt in enumerate(idxtgen(colorpools_rgb), 1):
                if 0 <= levelat_c:
                    if c == levelat_c:
                        levelat_c = -1
                    continue
                if c in {nextprintc, tcnt}:
                    printtime = datetime.datetime.now()
                    n = c - prevprintc
                    total_time += printtime - prevprinttime
                    total_cnt += n

                    estend = printtime + total_time / total_cnt * (tcnt-c)

                    print(f'{c} of {tcnt}, estimate end by {estend:%Y-%m-%d %H:%M:%S}', end="\r")

                    prevprinttime = printtime
                    prevprintc = c
                    nextprintc = c + random.randint(100, 500)

                rgb_palette = tuple(tuple(int(x) for x in colorpools_rgb[i][j]) for i, j in idxt)
                idxt_dE = (999999999,)
                for cv, lab_pool in cv_colorpools_lab.items():
                    lab_arr = np.array([lab_pool[i][j] for i, j in idxt], dtype="float64")
                    dE = de2000.get_pal_delta_e_from_lab_arr(lab_arr)[:-1]  # last black-white dE value ignored
                    if dE < idxt_dE:
                        idxt_dE = dE
                highs[idxt_dE].add(rgb_palette)
                highs_pop += 1

                if (highs_pop % 500000) == 0:  # save some memory
                    print("combinations at 500000; cutting back...", end="\r")
                    dEsorted = sorted(highs)
                    for dE in dEsorted:
                        c = len(highs[dE])
                        del highs[dE]
                        highs_pop -= c
                        if highs_pop <= 250000:
                            break

                if not c % 100000:
                    fp = cvdpaldir / f'cvd{cvd_n}.highs.{start_level}.{level}.at.{c}.txt'
                    cvdpaldir.mkdir(parents=True, exist_ok=True)
                    write_highs(highs, fp)
                    if prevatfp:
                        prevatfp.unlink()  # delete
                    prevatfp = fp

        except KeyboardInterrupt as e:
            if not highsfp:
                print()
            raise
        else:
            if not highsfp:
                print()

        if 250000 < highs_pop:  # save some memory
            print(f'combinations at {highs_pop}; final cutting back...', end="\r")
            dEsorted = sorted(highs)
            for dE in dEsorted:
                c = len(highs[dE])
                del highs[dE]
                highs_pop -= c
                if highs_pop <= 250000:
                    break
            print(f'{highs_pop} combinations were kept.')

    else:
        start_tcnt = idxtcnt(colorpools_rgbs[start_level])

    fp = cvdpaldir / f'cvd{cvd_n}.highs.{start_level}.{level}.txt'
    cvdpaldir.mkdir(parents=True, exist_ok=True)
    write_highs(highs, fp)

    dEsorted = sorted(highs, reverse=True)
    print(f'dE = {dEsorted[0][0]}')

    set_refpals = set()
    tr_refpals_all = []
    for dE in dEsorted:
        for pal_rgb in sorted(highs[dE]):
            refpal = tuple(rgbpyramid.get_ref_rgb(rgb, level, max255=True) for rgb in pal_rgb)
            refpal = refpal[1:-1]  # igonre black and white
            if refpal in set_refpals:
                continue
            tr_refpals_all.append((dE, refpal))
            set_refpals.add(refpal)
    del set_refpals

    prev_tcnt = start_tcnt

    while level < 256:

        old_level = level
        level *= 2
        new_level = level

        print(f'=== LEVEL: {level}')


        answer = "*** PLACEHOLDER ***"
        while answer:

            while not answer.isdecimal():
                print("Please give a target iteration count value for the next level")
                if prev_tcnt:
                    print(f'or hit enter for previous target {prev_tcnt}')
                print("or type Q or E for exit.")
                answer = input("::: ").lower()
                if answer in {"q", "e"}:
                    sys.exit(100)
                if not answer and prev_tcnt:
                    tcnt = prev_tcnt
                    break
                elif answer.isdecimal():
                    if len(tr_refpals_all) <= int(answer):
                        answer = ""
                        continue
                    tcnt = int(answer)

            tr_tcnts = []
            i0 = 0
            pick_i0 = None
            while i0 < len(tr_refpals_all) and sum(tr_tcnts) < tcnt:
                dE, tr_refpal = tr_refpals_all[i0]
                tr_colorpools_rgb_set = [set() for _ in range(cvd_n)]
                tr_colorpools_rgb_set[0] = {(0, 0, 0)}
                tr_colorpools_rgb_set[-1] = {(255, 255, 255)}
                #print(f'dE: {", ".join([format(x, ".3f") for x in dE])}')
                #print(f'REF: {palstr(tr_refpal)}')
                for i_color, refrgb0 in enumerate(tr_refpal, 1):
                    #print(i_color, refrgb0)
                    sameref_colors = set()
                    new_level_i_colors = colorpools_rgbs[new_level][i_color]
                    for rgb in new_level_i_colors:
                        rgb = tuple(int(x) for x in rgb)  # from numpy array to python
                        refrgb1 = rgbpyramid.get_ref_rgb(rgb, old_level, max255=True)
                        if refrgb1 == refrgb0:
                            sameref_colors.add(rgb)
                    #print(palstr(sameref_colors))
                    tr_colorpools_rgb_set[i_color] = sameref_colors
                tr_tcnt = idxtcnt(tr_colorpools_rgb_set)
                tr_tcnts.append(tr_tcnt)
                cum_tr_tcnt = sum(tr_tcnts)
                #print(f'i={i0}/{len(tr_refpals_all)}; dE2000={dE[0]:.3f}; count: {tr_tcnt}; cumulated: {cum_tr_tcnt}')
                if pick_i0 is None and start_tcnt < cum_tr_tcnt:
                    pick_i0 = i0
                    print(f'i={i0}/{len(tr_refpals_all)}; dE2000={dE[0]:.3f}; iteration count: {cum_tr_tcnt}')
                    print("^^^ CURRENTLY SELECTED ^^^")
                i0 += 1

            answer = ""
            while True:
                print("Please accept iteration count with Enter key or give a new value")
                print("or type Q or E for exit.")
                answer = input("::: ").lower()
                if answer in {"q", "e"}:
                    sys.exit(100)
                if not answer:
                    break
                elif answer.isdecimal():
                    if len(tr_refpals_all) <= int(answer):
                        continue
                    tcnt = int(answer)
                    break

        tcnt = sum(tr_tcnts[:pick_i0+1])
        cr = 0

        highs = collections.defaultdict(set)
        highs_pop = 0

        level_start_time = datetime.datetime.now()
        prevprinttime = level_start_time

        prevprintc = 0
        nextprintc = 1
        prevmsg = ""

        prevatfp = None

        for i0 in range(pick_i0 + 1):
            dE, tr_refpal = tr_refpals_all[i0]
            tr_colorpools_rgb_set = [set() for _ in range(cvd_n)]
            tr_colorpools_rgb_set[0] = {(0, 0, 0)}
            tr_colorpools_rgb_set[-1] = {(255, 255, 255)}
            for i_color, refrgb0 in enumerate(tr_refpal, 1):
                sameref_colors = set()
                new_level_i_colors = colorpools_rgbs[new_level][i_color]
                for rgb in new_level_i_colors:
                    rgb = tuple(int(x) for x in rgb)  # from numpy array to python
                    refrgb1 = rgbpyramid.get_ref_rgb(rgb, old_level, max255=True)
                    if refrgb1 == refrgb0:
                        sameref_colors.add(rgb)
                tr_colorpools_rgb_set[i_color] = sameref_colors
            tr_tcnt = idxtcnt(tr_colorpools_rgb_set)
            colorpools_rgb = [np.array(tuple(s),dtype="uint8") for s in tr_colorpools_rgb_set]

            cv_colorpools_rgb = {}
            cv_colorpools_lab = {}
            for cv, cvhaldclut in colviss.items():
                cv_colorpools_rgb[cv] = [cvhaldclut.clut[a[:,0], a[:,1], a[:,2]] for a in colorpools_rgb]
                cv_colorpools_lab[cv] = [de2000.get_lab_arr(rgb_arr) for rgb_arr in cv_colorpools_rgb[cv]]

            i0_start_time = datetime.datetime.now()

            #print(f'i={i0}/{pick_i0+1}; dE2000={dE[0]:.3f}; start time: {i0_start_time:%Y-%m-%d %H:%M:%S}')
            #if prevmsg:
            #    print(prevmsg, end="\r")

            try:
                for c_, idxt in enumerate(idxtgen(colorpools_rgb), 1):

                    c = cr + c_

                    #print(cr, c_, c, nextprintc, tcnt)

                    if c in {nextprintc, tcnt}:
                        printtime = datetime.datetime.now()
                        n = c - prevprintc
                        total_time += printtime - prevprinttime
                        total_cnt += n

                        estend = printtime + total_time / total_cnt * (tcnt-c)

                        prevmsg = f'i={i0}/{pick_i0+1}; dE2000={tr_refpals_all[i0][0][0]:.3f}; {c} of {tcnt}, estimate end by {estend:%Y-%m-%d %H:%M:%S}'
                        print(prevmsg, end="\r")

                        prevprinttime = printtime
                        prevprintc = c
                        nextprintc = c + random.randint(100, 500)

                    rgb_palette = tuple(tuple(int(x) for x in colorpools_rgb[i][j]) for i, j in idxt)
                    #print()
                    #print(palstr(rgb_palette))
                    #if prevmsg:
                    #    print(prevmsg, end="\r")
                    idxt_dE = (999999999,)
                    for cv, lab_pool in cv_colorpools_lab.items():
                        lab_arr = np.array([lab_pool[i][j] for i, j in idxt], dtype="float64")
                        dE = de2000.get_pal_delta_e_from_lab_arr(lab_arr)[:-1]  # last black-white dE value ignored
                        if dE < idxt_dE:
                            idxt_dE = dE
                    highs[idxt_dE].add(rgb_palette)
                    highs_pop += 1

                    if (highs_pop % 500000) == 0:  # save some memory
                        print("combinations at 500000; cutting back...", end="\r")
                        dEsorted = sorted(highs)
                        for dE in dEsorted:
                            c = len(highs[dE])
                            del highs[dE]
                            highs_pop -= c
                            if highs_pop <= 250000:
                                break

                    if not c % 100000:
                        fp = cvdpaldir / f'cvd{cvd_n}.highs.{start_level}.{level}.at.{c}.txt'
                        cvdpaldir.mkdir(parents=True, exist_ok=True)
                        write_highs(highs, fp)
                        if prevatfp:
                            prevatfp.unlink()  # delete
                        prevatfp = fp

            except KeyboardInterrupt as e:
                if not highsfp:
                    print()
                raise
            else:
                if not highsfp:
                    print()

            cr = c

        if 250000 < highs_pop:  # save some memory
            print(f'combinations at {highs_pop}; final cutting back...', end="\r")
            dEsorted = sorted(highs)
            for dE in dEsorted:
                c = len(highs[dE])
                del highs[dE]
                highs_pop -= c
                if highs_pop <= 250000:
                    break
            print(f'{highs_pop} combinations were kept.')

        fp = cvdpaldir / f'cvd{cvd_n}.highs.{start_level}.{level}.txt'
        cvdpaldir.mkdir(parents=True, exist_ok=True)
        write_highs(highs, fp)

        dEsorted = sorted(highs, reverse=True)
        print(f'dE = {dEsorted[0][0]}')

        set_refpals = set()
        tr_refpals_all = []
        for dE in dEsorted:
            for pal_rgb in sorted(highs[dE]):
                refpal = tuple(rgbpyramid.get_ref_rgb(rgb, level, max255=True) for rgb in pal_rgb)
                refpal = refpal[1:-1]  # igonre black and white
                if refpal in set_refpals:
                    continue
                tr_refpals_all.append((dE, refpal))
                set_refpals.add(refpal)
        del set_refpals

        prev_tcnt = tcnt
