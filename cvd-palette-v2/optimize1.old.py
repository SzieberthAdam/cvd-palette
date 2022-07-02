# standard libraries
import gc
import itertools
import pathlib
import sys

_thisdir = pathlib.Path(__file__).parent.resolve()
_importdir = _thisdir.parent.resolve()
sys.path.insert(0, str(_importdir))

# 3rd party libraries (module file available)
import clut
import de2000
import pal

sys.path.remove(str(_importdir))

# 3rd party libraries
import colour
from PIL import Image
import numpy as np


import sortpals


batchsize = 100000


clutschars = "NDPT"
clutnames = (
    "normal",
    "deuteranopia",
    "protanopia",
    "tritanopia",
)


def rgbarr_to_labarr(arr):
    shape = arr.shape
    arr = colour.sRGB_to_XYZ(arr/255)
    arr = colour.XYZ_to_Lab(arr).reshape(shape)
    return arr


# def rgbstr(rgb):
#     r, g, b = rgb
#     return f'#{r:0>2x}{g:0>2x}{b:0>2x}'


# def palstr(palarr):
#     return ", ".join(rgbstr(rgb) for rgb in palarr)


def grouper(iterable, n):
    args = [iter(iterable)] * n
    yield from (tuple(x for x in t if x is not ...) for t in itertools.zip_longest(*args, fillvalue=...))


def get_pal_values(pal_, cluts):
    n_vision = len(cluts)
    n_pals = 1  # technical to mimic main logic
    n_colors = pal_.shape[-2]
    b = np.array([[0] * n_colors])  # technical to mimic main logic
    combs = tuple(itertools.combinations(range(n_colors), 2))
    n_pairs = len(combs)
    color_arrs = [a.reshape((1, 3)) for a in pal_[0]]
    dE_arr = np.zeros((n_pals, n_pairs, n_vision), dtype="float64")
    for v, clut_ in enumerate(cluts):
        this_color_arrs = [clut_.clut[color_arrs[vp][:,0], color_arrs[vp][:,1], color_arrs[vp][:,2]] for vp in range(len(color_arrs))]
        for p, (c1i, c2i) in enumerate(combs):
            lab1 = rgbarr_to_labarr(this_color_arrs[c1i][b[:,c1i]])
            lab2 = rgbarr_to_labarr(this_color_arrs[c2i][b[:,c2i]])
            dE_arr[:,p][:,v] = this_dE_arr = de2000.delta_e_from_lab(lab1, lab2)
    sort1_dE_arr = np.sort(np.min(dE_arr, axis=1))
    sort2_dE_arr = np.sort(dE_arr.reshape((dE_arr.shape[0], dE_arr.shape[1]* dE_arr.shape[2])), axis=1)
    sort3_dE_arr = np.hstack((sort1_dE_arr, sort2_dE_arr))
    sort_ii = np.flip(np.lexsort(np.rot90(sort3_dE_arr)))  # technical to mimic main logic
    batch_best_i = sort_ii[0]  # technical to mimic main logic
    batch_best_dE = dE_arr[batch_best_i]
    batch_best_sorted_dE = sort3_dE_arr[batch_best_i]
    #batch_best_pal = np.asarray([tuple(color_arrs[c][i]) for c, i in enumerate(b[batch_best_i])]).reshape((1, -1, 3))
    return batch_best_dE, batch_best_sorted_dE


def dEstr(dE, combs, batchnr=None):
    ldE0 = np.min(dE, axis=0)
    vision_sort = tuple(np.argsort(ldE0))
    longest_clutname_len = max([len(s) for s in clutnames])
    _fs = f'{{0:<{longest_clutname_len}}}'
    rowheaders = [_fs.format(clutnames[i]) for i in vision_sort]
    colheaders = [f'{"lowest":^8}'] + [f'{str(t):^8}' for t in combs]
    tdE = np.transpose(dE)
    ldE = [ldE0[i] for i in vision_sort]
    dEvals = [[f'{ldE0[i]:>8.4f}'] + [f'{v:>8.4f}' for v in tdE[i]] for i in vision_sort]
    if batchnr is None:
        lines = []
    else:
        lines = [str(batchnr), ""]
    for li in range(len(rowheaders) + 2):
        if li == 0:
            s = " " * len(rowheaders[0]) + " | ".join([""] + colheaders)
            linelen = len(s)
        elif li == 1:
            s = "-" * linelen
        else:
            s = rowheaders[li-2] + " | ".join([""] + dEvals[li-2])
        lines.append(s)
    return "\n".join(lines+[""])


def get_color_optimization_order(dE_arr, n_colors, no_colors = None):
    combs = tuple(itertools.combinations(range(n_colors), 2))
    if len(dE_arr.shape) == 3:
        dE_arr = dE_arr[0]
    min_dE_by_pairs = np.min(dE_arr, axis=1) # shape = (n_pals, n_pairs)
    argsort_dE_arr_by_pairs = np.argsort(min_dE_by_pairs)  # shape = (n_pals, n_pairs)
    jj = []
    jjset = {0, n_colors-1}

    for i in argsort_dE_arr_by_pairs:
        for j in combs[i]:
            if j in jjset:
                continue
            else:
                jj.append(j)
                jjset.add(j)
    return tuple(x+1 for x in jj)

def optimize1(root, in_rgb_arr, color_nr = None, dEdist=10):
    haldclutdir = root.parent / "haldclut"
    cluts = (
        clut.CLUT(str(haldclutdir / "identity" / "identity.png")),       # normal
        clut.CLUT(str(haldclutdir / "cvd" / "deuta.machado2010.png")),   # deuteranopia
        clut.CLUT(str(haldclutdir / "cvd" / "prota.machado2010.png")),   # protanopia
        clut.CLUT(str(haldclutdir / "cvd" / "trita.machado2010.png")),   # tritanopia
    )
    n_vision = len(cluts)
    if len(in_rgb_arr.shape) == 2:
        in_rgb_arr = in_rgb_arr.reshape((1,) + in_rgb_arr.shape)
    not_close_to_rgb_arr = in_rgb_arr[:-1]
    rgb_arr = in_rgb_arr[-1:]  # optimize last
    not_close_to_lab_arr = rgbarr_to_labarr(not_close_to_rgb_arr)
    n_colors = in_rgb_arr.shape[1]
    combs = tuple(itertools.combinations(range(n_colors), 2))

    graypalroot = root.parent / "grayscale-palette"
    graypalp = graypalroot / f'grayscale{n_colors:0>2}.pal'
    with graypalp.open() as f:
        graypal = tuple(rgb[0] for rgb in pal.load(f))

    graypalstrs = tuple(f'0x{v:0>2X}' for v in graypal)
    print(f'Palette gray levels: {", ".join(graypalstrs)}')

    best_pal = rgb_arr = in_rgb_arr
    dE_arr = sortpals.get_dE_arr(rgb_arr)
    best_dE = dE_arr[0]
    best_sorted_dE = sortpals.argsort_rgb_arr_keys(rgb_arr, dE_arr=dE_arr)[0]

    batch_best_pal = None
    batch_best_dE = None
    batch_best_sorted_dE = None

    if color_nr is None:






if __name__ == "__main__":

    np.seterr(all='raise')

    root = pathlib.Path(__file__).parent.resolve()

    haldclutdir = root.parent / "haldclut"
    cluts = (
        clut.CLUT(str(haldclutdir / "identity" / "identity.png")),       # normal
        clut.CLUT(str(haldclutdir / "cvd" / "deuta.machado2010.png")),   # deuteranopia
        clut.CLUT(str(haldclutdir / "cvd" / "prota.machado2010.png")),   # protanopia
        clut.CLUT(str(haldclutdir / "cvd" / "trita.machado2010.png")),   # tritanopia
    )
    n_vision = len(cluts)

    usage = """Usage: python optimize1.py <palette image> <color number>"""
    try:
        in_img_path = pathlib.Path(sys.argv[1])
    except (IndexError, ValueError):
        print(usage)
        sys.exit(1)

    if 2 < len(sys.argv):
        try:
            color_nr = int(sys.argv[2])
        except (ValueError):
            print(usage)
            sys.exit(2)
    else:
        color_nr = None

    in_img = Image.open(str(in_img_path))
    in_img_rgb_arr = np.array(in_img, dtype="uint8")
    in_rgb_arr = in_img_rgb_arr[:1]  # keep first palette only
    n_colors = in_rgb_arr.shape[1]

    combs = tuple(itertools.combinations(range(n_colors), 2))
    #in_rgb_arr = in_img_rgb_arr.reshape((-1, 3))
    #in_lab_arr = rgbarr_to_labarr(in_rgb_arr)

    graypalroot = root.parent / "grayscale-palette"
    graypalp = graypalroot / f'grayscale{n_colors:0>2}.pal'
    with graypalp.open() as f:
        graypal = tuple(rgb[0] for rgb in pal.load(f))

    graypalstrs = tuple(f'0x{v:0>2X}' for v in graypal)
    print(f'Palette gray levels: {", ".join(graypalstrs)}')

    best_pal = rgb_arr = in_rgb_arr
    dE_arr = sortpals.get_dE_arr(rgb_arr)
    best_dE = dE_arr[0]
    best_sorted_dE = sortpals.argsort_rgb_arr_keys(rgb_arr, dE_arr=dE_arr)[0]

    batch_best_pal = None
    batch_best_dE = None
    batch_best_sorted_dE = None

    if color_nr == None:
        color_nr = get_color_optimization_order(dE_arr, n_colors, no_colors)[0]

    step = 0
    no_colors = set()
    while color_nrs and len(no_colors) < n_colors - 2:
        step += 1
        color_nr = color_nrs[0]
        if color_nr is ...:
            color_nr = get_color_optimization_order(dE_arr, n_colors, no_colors)[0]
            no_colors.add(color_nr)
        else:
            color_nrs = color_nrs[1:]
            no_colors = {color_nr}

        print(f'=== C{color_nr} ===')

        out_img_path = in_img_path.with_suffix(f'.opt1-{step:0>2}-{color_nr:0>2}' + in_img_path.suffix)
        out_dE_path = out_img_path.with_suffix(".txt")

        try:
            graylvl = graypal[color_nr-1]
        except (IndexError):
            print("Wrong color number.")
            print(usage)
            sys.exit(2)

        print(f'Gray level: 0x{graylvl:0>2X}')



        color_arrs = [None] * n_colors
        isoluminantdir = root.parent / "isoluminant"
        for i, a in enumerate(in_rgb_arr[0]):
            if i == color_nr-1:
                isoluminantimgpath = isoluminantdir / "szieberth" / f'{graylvl:0>2x}.png'
                assert isoluminantimgpath.is_file()
                img = Image.open(str(isoluminantimgpath))
                arr = np.array(img).reshape((-1, 3))
            else:
                arr = a.reshape((1, 3))
            color_arrs[i] = arr

        color_lab_arrs = [rgbarr_to_labarr(a) for a in color_arrs]
        iteration_count = int(np.prod([a.shape[0] for a in color_arrs], dtype="uint64"))
        print(f'Iterations: {iteration_count} {[a.shape[0] for a in color_arrs]!r}')
        idxtgen_args = [range(len(color_arrs[p])) for p in range(len(color_arrs))]
        idxtgen = itertools.product(*idxtgen_args)
        groupergen = grouper(idxtgen, batchsize)

        prevbatchnr = None

        for batchnr, batch in enumerate(groupergen, 1):
            if prevbatchnr and batchnr <= prevbatchnr:
                continue
            print(f'B{batchnr}.', end='', flush=True)
            b = np.asarray(batch)
            n_pals = len(batch)
            n_colors = len(batch[0])
            combs = tuple(itertools.combinations(range(n_colors), 2))
            n_pairs = len(combs)
            dE_arr = np.zeros((n_pals, n_pairs, n_vision), dtype="float64")
            print(":", end='', flush=True)
            for v, clut_ in enumerate(cluts):
                if v == 0:
                    this_color_arrs = color_arrs
                else:
                    this_color_arrs = [
                        clut_.clut[color_arrs[vp][:,0], color_arrs[vp][:,1], color_arrs[vp][:,2]]
                        for vp in range(len(color_arrs))
                    ]
                print(f'[{clutschars[v]}]', end="", flush=True)
                for p, (c1i, c2i) in enumerate(combs):
                    print(f'c{p}', end='', flush=True)
                    lab1 = rgbarr_to_labarr(this_color_arrs[c1i][b[:,c1i]])
                    lab2 = rgbarr_to_labarr(this_color_arrs[c2i][b[:,c2i]])
                    print("L", end="", flush=True)
                    dE_arr[:,p][:,v] = this_dE_arr = de2000.delta_e_from_lab(lab1, lab2)
                    print("D", end="", flush=True)

            sort_dE_arr = sortpals.argsort_rgb_arr_keys(None, dE_arr=dE_arr)
            # sort_ii = sortpals.argsort_rgb_arr(None, dE_arr=dE_arr) # identical to next line
            sort_ii = np.flip(np.lexsort(np.rot90(sort_dE_arr)))  # shape = (n_pals,)

            ii = sort_ii[0]

            pal_rgb_arr = np.asarray([tuple(color_arrs[c][i]) for c, i in enumerate(b[ii])]).reshape((1, -1, 3))
            pal_lab_arr = np.asarray([tuple(color_lab_arrs[c][i]) for c, i in enumerate(b[ii])]).reshape((1, -1, 3))


            batch_best_i = ii # !!!
            batch_best_pal = np.asarray([tuple(color_arrs[c][i]) for c, i in enumerate(b[batch_best_i])]).reshape((1, -1, 3))
            batch_best_dE = dE_arr[batch_best_i]
            batch_best_sorted_dE = sort_dE_arr[batch_best_i]
            if best_sorted_dE is None or (tuple(best_sorted_dE) < tuple(batch_best_sorted_dE)):
                print("!!!", end="", flush=True)
                best_dE = batch_best_dE
                best_sorted_dE = batch_best_sorted_dE
                best_pal = batch_best_pal
                img = Image.fromarray(best_pal, 'RGB')
                img.save(out_img_path)
                print("(I)", end="", flush=True)
                no_colors = {color_nr}
            with out_dE_path.open("w") as f:
                f.write(dEstr(best_dE, combs, batchnr))
            print("(B)", end="", flush=True)
            print(flush=True)
        else:
            with out_dE_path.open("w") as f:
                f.write(dEstr(best_dE, combs))
            print("E", flush=True)
        out_rgb_arr = best_pal
        # out_rgb_arr = sortpals.sort_rgb_arr(out_rgb_arr)
        out_lab_arr = rgbarr_to_labarr(out_rgb_arr)
        out_img = Image.fromarray(out_rgb_arr, 'RGB')
        out_img.save(out_img_path)
        report_str = sortpals.report_str(out_rgb_arr)
        with out_dE_path.open("w", encoding="utf8", newline='\r\n') as f:
            f.write(report_str)
