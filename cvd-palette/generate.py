# standard libraries
import bisect
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

PALETTE_SIZE = 4096

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

def grouper(iterable, n):
    args = [iter(iterable)] * n
    yield from (tuple(x for x in t if x is not ...) for t in itertools.zip_longest(*args, fillvalue=...))

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


def write_palette(palettes, filestem):
    sorted_palettes = tuple(t for t in reversed(palettes))
    de_file = pathlib.Path(filestem).with_suffix(".dE.txt")
    lines = tuple(
        f'{i:>4}: '
        + " ".join([f'{v:>8.4f}' for v in de])
        for i, (de, palette) in enumerate(sorted_palettes)
    )
    with de_file.open("w") as f:
        f.write("\n".join( lines + ("",) ))
    pallist_file = pathlib.Path(filestem).with_suffix(".pal.txt")
    lines = tuple(
        f'{i:>4}: '
        + palstr(palette)
        for i, (de, palette) in enumerate(sorted_palettes)
    )
    with pallist_file.open("w") as f:
        f.write("\n".join( lines + ("",) ))
    png_file = pathlib.Path(filestem).with_suffix(".pal.png")
    n_colors = len(sorted_palettes[0][1])
    img_arr = np.array(
        tuple(palette for de, palette in sorted_palettes),
        dtype="uint8"
    ).reshape((len(sorted_palettes), n_colors, 3))
    img = Image.fromarray(img_arr, 'RGB')
    img.save(str(png_file))


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

    try:
        cvd_n = int(sys.argv[1])
    except (IndexError, ValueError):
        print("Usage: python generate.py <palette size>")
        sys.exit(1)

    np.seterr(all='raise')

    root = pathlib.Path(__file__).parent.resolve()

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

    graypalstrs = tuple(f'0x{v:0>2X}' for v in graypal)
    print(f'Palette gray levels: {", ".join(graypalstrs)}')

    levels = tuple(range(len(rgbpyramid.LEVEL)))

    level_isoluminant_color_count = {}
    for level_ in levels:

        leveldir = root.parent / "isoluminant" / f'level{level_}'
        level_isoluminant_color_count[level_] = colorpools_rgb = []
        for v in graypal:
            if v in {0, 255}:
                colorpools_rgb.append(np.array(((v, v, v),), dtype="uint8"))
            else:
                try:
                    img = Image.open(str(leveldir / f'{v:0>2x}.png'))
                except FileNotFoundError:
                    colorpools_rgb.append(np.array((), dtype="uint8"))
                else:
                    arr = np.array(img).reshape((-1, 3))
                    colorpools_rgb.append(arr)
        cnt = idxtcnt(colorpools_rgb)
        print(f'Palette count for start level {level_} = {cnt}')

    answer = ""
    while answer not in {str(x) for x in levels}:
        answer = input("Level (or Q to quit): ").lower()
        if answer in {"q", "e"}:
            sys.exit(100)
    level = start_level = int(answer)


    ## start

    cvdpaldir = root / f'cvd{cvd_n}'
    cvdpaldir.mkdir(parents=True, exist_ok=True)

    filestem = cvdpaldir / f'cvd{cvd_n}-lvl{start_level}-{level}'
    png_file = pathlib.Path(filestem).with_suffix(".pal.png")

    cv_colorpools_rgb = {}
    cv_colorpools_lab = {}

    if not png_file.is_file():

        colorpools_rgb = level_isoluminant_color_count[level]

        for cv, cvhaldclut in colviss.items():
            cv_colorpools_rgb[cv] = [cvhaldclut.clut[a[:,0], a[:,1], a[:,2]] for a in colorpools_rgb]
            cv_colorpools_lab[cv] = [de2000.get_lab_arr(rgb_arr) for rgb_arr in cv_colorpools_rgb[cv]]

        palettes = []

        for c, idxt in enumerate(idxtgen(colorpools_rgb)):
            palette = tuple(tuple(int(x) for x in colorpools_rgb[i][j]) for i, j in idxt)
            idxt_dE = (999999999,)
            for cv, lab_pool in cv_colorpools_lab.items():
                lab_arr = np.array([lab_pool[i][j] for i, j in idxt], dtype="float64")
                dE = de2000.get_pal_delta_e_from_lab_arr(lab_arr)
                if dE < idxt_dE:
                    idxt_dE = dE
            t = (idxt_dE, palette)
            bisect.insort_right(palettes, t)

            if c % 10000 == 0:
                print(c)
                palettes = palettes[-PALETTE_SIZE:]

        palettes = palettes[-PALETTE_SIZE:]
        write_palette(palettes, filestem)

    while 0 < level:
        prev_filestem = filestem
        prev_png_file = png_file
        level -= 1
        filestem = cvdpaldir / f'cvd{cvd_n}-lvl{start_level}-{level}'
        png_file = pathlib.Path(filestem).with_suffix(".pal.png")

        if not png_file.is_file():

            prev_img = Image.open(str(prev_png_file))
            prev_arr = np.asarray(prev_img)

            colorpools_rgb = level_isoluminant_color_count[level]
            colorpool_idxt = {}
            for i, colorpool_rgb in enumerate(colorpools_rgb):
                for j, rgb in enumerate(colorpool_rgb):
                    colorpool_idxt[tuple(rgb)] = i, j
            for cv, cvhaldclut in colviss.items():
                cv_colorpools_rgb[cv] = [cvhaldclut.clut[a[:,0], a[:,1], a[:,2]] for a in colorpools_rgb]
                cv_colorpools_lab[cv] = [de2000.get_lab_arr(rgb_arr) for rgb_arr in cv_colorpools_rgb[cv]]

            palettes = []

            tot = 0
            p = 0
            for p, source_palette in enumerate(prev_arr):
                gen = rgbpyramid.iterpalettes(level+1, source_palette, val00fffixed=True)
                c = 0
                for palette in gen:
                    try:
                        idxt = tuple(
                            colorpool_idxt[rgb]
                            for i, rgb in enumerate(palette)
                        )
                    except KeyError:
                        continue
                    idxt_dE = (999999999,)
                    for cv, lab_pool in cv_colorpools_lab.items():
                        lab_arr = np.array([lab_pool[i][j] for i, j in idxt], dtype="float64")
                        dE = de2000.get_pal_delta_e_from_lab_arr(lab_arr)
                        if dE < idxt_dE:
                            idxt_dE = dE
                    t = (idxt_dE, palette)
                    bisect.insort_right(palettes, t)
                    c += 1
                    tot += 1

                    if tot % 10000 == 0:
                        print(f'{tot}: palette.{p}; child.{c}')
                        palettes = palettes[-PALETTE_SIZE:]

            palettes = palettes[-PALETTE_SIZE:]
            write_palette(palettes, filestem)









#    total_time = datetime.timedelta(0)
#    total_cnt = 0
#
#    if not highsfp:
#
#        # exceptional first pass
#
#        print(f'=== START LEVEL: {level}')
#
#        if not highsfp:
#            highs = collections.defaultdict(set)
#            highs_pop = 0
#
#        cv_colorpools_rgb = {}
#        cv_colorpools_lab = {}
#        for cv, cvhaldclut in colviss.items():
#            cv_colorpools_rgb[cv] = [cvhaldclut.clut[a[:,0], a[:,1], a[:,2]] for a in colorpools_rgb]
#            cv_colorpools_lab[cv] = [de2000.get_lab_arr(rgb_arr) for rgb_arr in cv_colorpools_rgb[cv]]
#
#        start_tcnt = tcnt = idxtcnt(colorpools_rgb)
#
#        start_level_start_time = level_start_time = datetime.datetime.now()
#        print(f'start time: {level_start_time:%Y-%m-%d %H:%M:%S}')
#
#        prevprinttime = level_start_time
#        prevprintc = 0
#        nextprintc = 1
#
#        prevatfp = None
#
#        try:
#            for c, idxt in enumerate(idxtgen(colorpools_rgb), 1):
#                if 0 <= levelat_c:
#                    if c == levelat_c:
#                        levelat_c = -1
#                    continue
#                if c in {nextprintc, tcnt}:
#                    printtime = datetime.datetime.now()
#                    n = c - prevprintc
#                    total_time += printtime - prevprinttime
#                    total_cnt += n
#
#                    estend = printtime + total_time / total_cnt * (tcnt-c)
#
#                    print(f'{c} of {tcnt}, estimate end by {estend:%Y-%m-%d %H:%M:%S}', end="\r")
#
#                    prevprinttime = printtime
#                    prevprintc = c
#                    nextprintc = c + random.randint(100, 500)
#
#                palette = tuple(tuple(int(x) for x in colorpools_rgb[i][j]) for i, j in idxt)
#                idxt_dE = (999999999,)
#                for cv, lab_pool in cv_colorpools_lab.items():
#                    lab_arr = np.array([lab_pool[i][j] for i, j in idxt], dtype="float64")
#                    dE = de2000.get_pal_delta_e_from_lab_arr(lab_arr)[:-1]  # last black-white dE value ignored
#                    if dE < idxt_dE:
#                        idxt_dE = dE
#                highs[idxt_dE].add(palette)
#                highs_pop += 1
#
#                if (highs_pop % 500000) == 0:  # save some memory
#                    print("combinations at 500000; cutting back...", end="\r")
#                    dEsorted = sorted(highs)
#                    for dE in dEsorted:
#                        c = len(highs[dE])
#                        del highs[dE]
#                        highs_pop -= c
#                        if highs_pop <= 250000:
#                            break
#
#                if not c % 100000:
#                    fp = cvdpaldir / f'cvd{cvd_n}.highs.{start_level}.{level}.at.{c}.txt'
#                    cvdpaldir.mkdir(parents=True, exist_ok=True)
#                    write_highs(highs, fp)
#                    if prevatfp:
#                        prevatfp.unlink()  # delete
#                    prevatfp = fp
#
#        except KeyboardInterrupt as e:
#            if not highsfp:
#                print()
#            raise
#        else:
#            if not highsfp:
#                print()
#
#        if 250000 < highs_pop:  # save some memory
#            print(f'combinations at {highs_pop}; final cutting back...', end="\r")
#            dEsorted = sorted(highs)
#            for dE in dEsorted:
#                c = len(highs[dE])
#                del highs[dE]
#                highs_pop -= c
#                if highs_pop <= 250000:
#                    break
#            print(f'{highs_pop} combinations were kept.')
#
#    else:
#        start_tcnt = idxtcnt(colorpools_rgbs[start_level])
#
#    fp = cvdpaldir / f'cvd{cvd_n}.highs.{start_level}.{level}.txt'
#    cvdpaldir.mkdir(parents=True, exist_ok=True)
#    write_highs(highs, fp)
#
#    dEsorted = sorted(highs, reverse=True)
#    print(f'dE = {dEsorted[0][0]}')
#
#    set_refpals = set()
#    tr_refpals_all = []
#    for dE in dEsorted:
#        for pal_rgb in sorted(highs[dE]):
#            refpal = tuple(rgbpyramid.get_ref_rgb(rgb, level, max255=True) for rgb in pal_rgb)
#            refpal = refpal[1:-1]  # igonre black and white
#            if refpal in set_refpals:
#                continue
#            tr_refpals_all.append((dE, refpal))
#            set_refpals.add(refpal)
#    del set_refpals
#
#    prev_tcnt = start_tcnt
#
#    while level < 256:
#
#        old_level = level
#        level *= 2
#        new_level = level
#
#        print(f'=== LEVEL: {level}')
#
#
#        answer = "*** PLACEHOLDER ***"
#        while answer:
#
#            while not answer.isdecimal():
#                print("Please give a target iteration count value for the next level")
#                if prev_tcnt:
#                    print(f'or hit enter for previous target {prev_tcnt}')
#                print("or type Q or E for exit.")
#                answer = input("::: ").lower()
#                if answer in {"q", "e"}:
#                    sys.exit(100)
#                if not answer and prev_tcnt:
#                    tcnt = prev_tcnt
#                    break
#                elif answer.isdecimal():
#                    if len(tr_refpals_all) <= int(answer):
#                        answer = ""
#                        continue
#                    tcnt = int(answer)
#
#            tr_tcnts = []
#            i0 = 0
#            pick_i0 = None
#            while i0 < len(tr_refpals_all) and sum(tr_tcnts) < tcnt:
#                dE, tr_refpal = tr_refpals_all[i0]
#                tr_colorpools_rgb_set = [set() for _ in range(cvd_n)]
#                tr_colorpools_rgb_set[0] = {(0, 0, 0)}
#                tr_colorpools_rgb_set[-1] = {(255, 255, 255)}
#                #print(f'dE: {", ".join([format(x, ".3f") for x in dE])}')
#                #print(f'REF: {palstr(tr_refpal)}')
#                for i_color, refrgb0 in enumerate(tr_refpal, 1):
#                    #print(i_color, refrgb0)
#                    sameref_colors = set()
#                    new_level_i_colors = colorpools_rgbs[new_level][i_color]
#                    for rgb in new_level_i_colors:
#                        rgb = tuple(int(x) for x in rgb)  # from numpy array to python
#                        refrgb1 = rgbpyramid.get_ref_rgb(rgb, old_level, max255=True)
#                        if refrgb1 == refrgb0:
#                            sameref_colors.add(rgb)
#                    #print(palstr(sameref_colors))
#                    tr_colorpools_rgb_set[i_color] = sameref_colors
#                tr_tcnt = idxtcnt(tr_colorpools_rgb_set)
#                tr_tcnts.append(tr_tcnt)
#                cum_tr_tcnt = sum(tr_tcnts)
#                #print(f'i={i0}/{len(tr_refpals_all)}; dE2000={dE[0]:.3f}; count: {tr_tcnt}; cumulated: {cum_tr_tcnt}')
#                if pick_i0 is None and start_tcnt < cum_tr_tcnt:
#                    pick_i0 = i0
#                    print(f'i={i0}/{len(tr_refpals_all)}; dE2000={dE[0]:.3f}; iteration count: {cum_tr_tcnt}')
#                    print("^^^ CURRENTLY SELECTED ^^^")
#                i0 += 1
#
#            answer = ""
#            while True:
#                print("Please accept iteration count with Enter key or give a new value")
#                print("or type Q or E for exit.")
#                answer = input("::: ").lower()
#                if answer in {"q", "e"}:
#                    sys.exit(100)
#                if not answer:
#                    break
#                elif answer.isdecimal():
#                    if len(tr_refpals_all) <= int(answer):
#                        continue
#                    tcnt = int(answer)
#                    break
#
#        tcnt = sum(tr_tcnts[:pick_i0+1])
#        cr = 0
#
#        highs = collections.defaultdict(set)
#        highs_pop = 0
#
#        level_start_time = datetime.datetime.now()
#        prevprinttime = level_start_time
#
#        prevprintc = 0
#        nextprintc = 1
#        prevmsg = ""
#
#        prevatfp = None
#
#        for i0 in range(pick_i0 + 1):
#            dE, tr_refpal = tr_refpals_all[i0]
#            tr_colorpools_rgb_set = [set() for _ in range(cvd_n)]
#            tr_colorpools_rgb_set[0] = {(0, 0, 0)}
#            tr_colorpools_rgb_set[-1] = {(255, 255, 255)}
#            for i_color, refrgb0 in enumerate(tr_refpal, 1):
#                sameref_colors = set()
#                new_level_i_colors = colorpools_rgbs[new_level][i_color]
#                for rgb in new_level_i_colors:
#                    rgb = tuple(int(x) for x in rgb)  # from numpy array to python
#                    refrgb1 = rgbpyramid.get_ref_rgb(rgb, old_level, max255=True)
#                    if refrgb1 == refrgb0:
#                        sameref_colors.add(rgb)
#                tr_colorpools_rgb_set[i_color] = sameref_colors
#            tr_tcnt = idxtcnt(tr_colorpools_rgb_set)
#            colorpools_rgb = [np.array(tuple(s),dtype="uint8") for s in tr_colorpools_rgb_set]
#
#            cv_colorpools_rgb = {}
#            cv_colorpools_lab = {}
#            for cv, cvhaldclut in colviss.items():
#                cv_colorpools_rgb[cv] = [cvhaldclut.clut[a[:,0], a[:,1], a[:,2]] for a in colorpools_rgb]
#                cv_colorpools_lab[cv] = [de2000.get_lab_arr(rgb_arr) for rgb_arr in cv_colorpools_rgb[cv]]
#
#            i0_start_time = datetime.datetime.now()
#
#            #print(f'i={i0}/{pick_i0+1}; dE2000={dE[0]:.3f}; start time: {i0_start_time:%Y-%m-%d %H:%M:%S}')
#            #if prevmsg:
#            #    print(prevmsg, end="\r")
#
#            try:
#                for c_, idxt in enumerate(idxtgen(colorpools_rgb), 1):
#
#                    c = cr + c_
#
#                    #print(cr, c_, c, nextprintc, tcnt)
#
#                    if c in {nextprintc, tcnt}:
#                        printtime = datetime.datetime.now()
#                        n = c - prevprintc
#                        total_time += printtime - prevprinttime
#                        total_cnt += n
#
#                        estend = printtime + total_time / total_cnt * (tcnt-c)
#
#                        prevmsg = f'i={i0}/{pick_i0+1}; dE2000={tr_refpals_all[i0][0][0]:.3f}; {c} of {tcnt}, estimate end by {estend:%Y-%m-%d %H:%M:%S}'
#                        print(prevmsg, end="\r")
#
#                        prevprinttime = printtime
#                        prevprintc = c
#                        nextprintc = c + random.randint(100, 500)
#
#                    palette = tuple(tuple(int(x) for x in colorpools_rgb[i][j]) for i, j in idxt)
#                    #print()
#                    #print(palstr(palette))
#                    #if prevmsg:
#                    #    print(prevmsg, end="\r")
#                    idxt_dE = (999999999,)
#                    for cv, lab_pool in cv_colorpools_lab.items():
#                        lab_arr = np.array([lab_pool[i][j] for i, j in idxt], dtype="float64")
#                        dE = de2000.get_pal_delta_e_from_lab_arr(lab_arr)[:-1]  # last black-white dE value ignored
#                        if dE < idxt_dE:
#                            idxt_dE = dE
#                    highs[idxt_dE].add(palette)
#                    highs_pop += 1
#
#                    if (highs_pop % 500000) == 0:  # save some memory
#                        print("combinations at 500000; cutting back...", end="\r")
#                        dEsorted = sorted(highs)
#                        for dE in dEsorted:
#                            c = len(highs[dE])
#                            del highs[dE]
#                            highs_pop -= c
#                            if highs_pop <= 250000:
#                                break
#
#                    if not c % 100000:
#                        fp = cvdpaldir / f'cvd{cvd_n}.highs.{start_level}.{level}.at.{c}.txt'
#                        cvdpaldir.mkdir(parents=True, exist_ok=True)
#                        write_highs(highs, fp)
#                        if prevatfp:
#                            prevatfp.unlink()  # delete
#                        prevatfp = fp
#
#            except KeyboardInterrupt as e:
#                if not highsfp:
#                    print()
#                raise
#            else:
#                if not highsfp:
#                    print()
#
#            cr = c
#
#        if 250000 < highs_pop:  # save some memory
#            print(f'combinations at {highs_pop}; final cutting back...', end="\r")
#            dEsorted = sorted(highs)
#            for dE in dEsorted:
#                c = len(highs[dE])
#                del highs[dE]
#                highs_pop -= c
#                if highs_pop <= 250000:
#                    break
#            print(f'{highs_pop} combinations were kept.')
#
#        fp = cvdpaldir / f'cvd{cvd_n}.highs.{start_level}.{level}.txt'
#        cvdpaldir.mkdir(parents=True, exist_ok=True)
#        write_highs(highs, fp)
#
#        dEsorted = sorted(highs, reverse=True)
#        print(f'dE = {dEsorted[0][0]}')
#
#        set_refpals = set()
#        tr_refpals_all = []
#        for dE in dEsorted:
#            for pal_rgb in sorted(highs[dE]):
#                refpal = tuple(rgbpyramid.get_ref_rgb(rgb, level, max255=True) for rgb in pal_rgb)
#                refpal = refpal[1:-1]  # igonre black and white
#                if refpal in set_refpals:
#                    continue
#                tr_refpals_all.append((dE, refpal))
#                set_refpals.add(refpal)
#        del set_refpals
#
#        prev_tcnt = tcnt
