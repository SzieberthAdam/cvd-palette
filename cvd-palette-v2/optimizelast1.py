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


import sortall


batchsize = 100000

haldclutdir = _thisdir.parent / "haldclut"
cluts = (
    clut.CLUT(str(haldclutdir / "identity" / "identity.png")),       # normal
    clut.CLUT(str(haldclutdir / "cvd" / "deuta.machado2010.png")),   # deuteranopia
    clut.CLUT(str(haldclutdir / "cvd" / "prota.machado2010.png")),   # protanopia
    clut.CLUT(str(haldclutdir / "cvd" / "trita.machado2010.png")),   # tritanopia
)
n_vision = len(cluts)
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


def rgbstr(rgb):
    r, g, b = rgb
    return f'#{r:0>2x}{g:0>2x}{b:0>2x}'


def palstr(palarr):
    return ", ".join(rgbstr(rgb) for rgb in palarr)

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


def get_color_optimization_order(dE_arr, n_colors):
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


def main(in_rgb_arr, min_nci_dE=10.0):
    optimized = False
    not_close_to_rgb_arr = in_rgb_arr[:-1]
    not_close_to_lab_arr = rgbarr_to_labarr(not_close_to_rgb_arr)
    rgb_arr = in_rgb_arr[-1:]  # optimize last
    print(palstr(rgb_arr[0]))
    n_colors = rgb_arr.shape[1]

    combs = tuple(itertools.combinations(range(n_colors), 2))

    graypalroot = _thisdir.parent / "grayscale-palette"
    graypalp = graypalroot / f'grayscale{n_colors:0>2}.pal'
    with graypalp.open() as f:
        graypal = tuple(rgb[0] for rgb in pal.load(f))

    graypalstrs = tuple(f'0x{v:0>2X}' for v in graypal)
    print(f'Palette gray levels: {", ".join(graypalstrs)}')

    step = 0
    last_step_color_nr = None
    while True:

        found_better = False

        dE_arr = sortall.get_dE_arr(rgb_arr)
        best_dE = dE_arr[0]
        best_sorted_dE = sortall.argsort_rgb_arr_keys(rgb_arr, dE_arr=dE_arr)[0]

        batch_best_pal = None
        batch_best_dE = None
        batch_best_sorted_dE = None

        color_nrs = get_color_optimization_order(dE_arr, n_colors)

        if last_step_color_nr == color_nrs[0]:
            color_nrs = color_nrs[1:]
        print(color_nrs)

        for color_nr in color_nrs:

            last_step_color_nr = color_nr
            step += 1
            found_better = False

            print(f'=== C{color_nr} ===')
            graylvl = graypal[color_nr-1]
            print(f'Gray level: 0x{graylvl:0>2X}')
            color_arrs = [None] * n_colors
            isoluminantdir = _thisdir.parent / "isoluminant"
            for i, a in enumerate(rgb_arr[0]):
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
            for batchnr, batch in enumerate(groupergen, 1):
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
                sort_dE_arr = sortall.argsort_rgb_arr_keys(None, dE_arr=dE_arr)
                # sort_ii = sortall.argsort_rgb_arr(None, dE_arr=dE_arr) # identical to next line
                sort_ii = np.flip(np.lexsort(np.rot90(sort_dE_arr)))  # shape = (n_pals,)
                if not_close_to_lab_arr.shape[0]:
                    for ii in sort_ii:
                        pal_rgb_arr = np.asarray([tuple(color_arrs[c][i]) for c, i in enumerate(b[ii])]).reshape((1, -1, 3))
                        pal_lab_arr = np.asarray([tuple(color_lab_arrs[c][i]) for c, i in enumerate(b[ii])]).reshape((1, -1, 3))
                        pal_nci_dE_arr = de2000.delta_e_from_lab(pal_lab_arr, not_close_to_lab_arr)
                        pal_nci_dE_max_arr = np.max(pal_nci_dE_arr, axis=1)
                        pal_nci_dE = np.min(pal_nci_dE_max_arr)
                        if min_nci_dE <= pal_nci_dE:
                            break
                    else:
                        continue
                else:
                    ii = sort_ii[0]
                batch_best_i = ii # !!!
                batch_best_pal = np.asarray([tuple(color_arrs[c][i]) for c, i in enumerate(b[batch_best_i])]).reshape((1, -1, 3))
                batch_best_dE = dE_arr[batch_best_i]
                batch_best_sorted_dE = sort_dE_arr[batch_best_i]
                if best_sorted_dE is None or (tuple(best_sorted_dE) < tuple(batch_best_sorted_dE)):
                    print("!!!", end="", flush=True)
                    best_dE = batch_best_dE
                    best_sorted_dE = batch_best_sorted_dE
                    rgb_arr = best_pal = batch_best_pal
                    found_better = True
                    optimized = True
                print("(B)", end="", flush=True)
                print(flush=True)
            else:
                print("E", flush=True)
            out_rgb_arr = np.vstack((not_close_to_rgb_arr, rgb_arr))

            if found_better:
                print(palstr(rgb_arr[0]))
                break

        else:  # not found better
            assert not found_better
            break # from while True

    return optimized, out_rgb_arr

if __name__ == "__main__":
    usage = """Usage: python optimizelast1.py <palette image> [min difference]"""
    try:
        in_img_path = pathlib.Path(sys.argv[1])
    except (IndexError, ValueError):
        print(usage)
        sys.exit(1)

    if 2 < len(sys.argv):
        try:
            min_nci_dE = float(sys.argv[2])
        except (ValueError):
            print(usage)
            sys.exit(2)
    else:
        min_nci_dE = 10.0

    in_img = Image.open(str(in_img_path))
    in_rgb_arr = np.array(in_img, dtype="uint8")

    optimized, out_rgb_arr = main(in_rgb_arr, min_nci_dE)

    out_img_path = in_img_path.with_suffix(f'.opt1' + in_img_path.suffix)
    out_dE_path = out_img_path.with_suffix(".txt")

    out_img = Image.fromarray(out_rgb_arr, 'RGB')
    out_img.save(out_img_path)
    report_str = sortall.report_str(out_rgb_arr)
    with out_dE_path.open("w", encoding="utf8", newline='\r\n') as f:
        f.write(report_str)
