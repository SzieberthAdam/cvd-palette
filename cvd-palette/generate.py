# standard libraries
import collections
from datetime import datetime
import itertools
import json
import pathlib
import sys

_thisdir = pathlib.Path(__file__).parent.resolve()
_importdir = _thisdir.parent.resolve()
sys.path.insert(0, str(_importdir))

# 3rd party libraries (module file available)
import clut
import pal

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
    haldclutdir = root.parent / "haldclut"

    colviss = {
        "normal": haldclutdir / "identity" / "identity.png",
        "deuteranopia": haldclutdir / "cvd" / "deuta.machado2010.png",
        "protanopia": haldclutdir / "cvd" / "prota.machado2010.png",
        "tritanopia": haldclutdir / "cvd" / "trita.machado2010.png",
    }

    graypalroot = root.parent / "grayscale-palette"
    graypalp = graypalroot / f'grayscale{cvd_n:0>2}.pal'
    with graypalp.open() as f:
        graypal = tuple(rgb[0] for rgb in pal.load(f))

    midgrays = tuple(graypal[1:-1])
    isoluminant_fnames = {v: f'{v:x}.png' for v in midgrays}

    level = 2
    leveldir = root.parent / "isoluminant" / f'level{level:0>3}'

    colors = []
    for v in graypal:
        if v in {0, 255}:
            colors.append((v, v, v))
        else:
            img = Image.open(str(leveldir / isoluminant_fnames[v]))
            arr = np.array(img)
            break



#    fpath_palgraylvls = root/"graylevel_palgraylvls.json"
#    with fpath_palgraylvls.open("r") as f:
#        graylevel_palgraylvls = json.load(f)
#    graylvls = tuple(graylevel_palgraylvls[f'cvd{CVD}.graylvls'])
#
#    if CVD == 3:
#        dirname0 = root / "achromatopsiacolorsbygraylvl/full"
#    elif CVD == 4:
#        dirname0 = root / "achromatopsiacolorsbygraylvl/full"
#    #dirname0 = root / "achromatopsiacolorsbygraylvl/256color_ct"
#    else:
#        dirname0 = root / "achromatopsiacolorsbygraylvl/32color_ct"
#    dirname1 = root / f'cvd{CVD}'
#    dirname1.mkdir(parents=True, exist_ok=True)  # ensure directory
#
#    labmap = keydefaultdict(lambda key: rgb_to_lab(key))
#    #deltaEmap = keydefaultdict(lambda key: delta_e(*key))
#
#    eh = {}
#    for cvdkey in cvds:
#        p = root / f'haldclut/{cvdkey}.png'
#        eh[cvdkey] = clut.CLUT(str(p))
#
#    graylvl_palrgbs = {0x00: [(0, 0, 0)], 0xFF: [(255, 255, 255)]}
#    graylvl_palarr = {}
#    graylvl_pallabarr = {}
#    graylvl_palimg = {}
#    for graylvl in graylvls:
#        if graylvl in (0x00, 0xff):
#            continue
#        img = Image.open(str(dirname0 / f'{graylvl:0>2x}.png'))
#        graylvl_palimg[graylvl] = img
#        arr = np.asarray(img, dtype=np.uint8)
#        graylvl_palarr[graylvl] = arr
#        graylvl_pallabarr[graylvl] = rgbarr_to_labarr(arr)
#        graylvl_palrgbs[graylvl] = list(tuple(a.tolist()) for a in arr[0])
#
#    sorted_graylvl_palrgbs = tuple(sorted(graylvl_palrgbs))
#    it = itertools.product(*[graylvl_palrgbs[graylvl] for graylvl in sorted_graylvl_palrgbs])
#
#    max_dE = max_dE or graylevel_palgraylvls[f'cvd{CVD}.graylvls.deltaE']
#    max_dE_pals = []
#    for i, pal in enumerate(it):
#        if not i % 100000:
#            print(f'i = {i}')
#        palarr = np.array([pal], dtype=np.uint8) # [pal] makes it 1 x n x 3 size
#        pal_dE = 1000
#        for cvdtyp in cvds:
#            cvd_dE = 1000
#            cvdpalarr = eh[cvdtyp](palarr)
#            cvdlabs = [labmap[tuple(rgb)] for rgb in cvdpalarr[0]]
#            for labpair in itertools.combinations(cvdlabs, 2):
#                #dE = deltaEmap[labpair]
#                dE = delta_e(*key)
#                cvd_dE = min(cvd_dE, dE)
#            pal_dE = min(pal_dE, cvd_dE)
#        if max_dE < pal_dE:
#            max_dE_pals = []
#            print()
#            print(f'{pal_dE}')
#        if max_dE <= pal_dE:
#            max_dE = pal_dE
#            max_dE_pals.append(pal)
#            palstr_ = palstr(pal)
#            print(palstr_)
#            with (dirname1 / f'{pal_dE}.txt').open("a") as f:
#                f.write("\n" + palstr_)
#
#    with (dirname1 / f'deltae.txt').open("w") as f:
#        json.dump(max_dE, f, **json_dump_kwargs)
#    with (dirname1 / f'palettes.json').open("w") as f:
#        json.dump(max_dE_pals, f, **json_dump_kwargs)
#    with (dirname1 / f'palettes.txt').open("w") as f:
#        f.write("\n".join(palstr(pal) for pal in max_dE_pals))
