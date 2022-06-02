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


batchsize = 100000


clutschars = "NDPT"
clutnames = (
    "normal",
    "deuteranopia",
    "protanopia",
    "tritanopia",
)


def rgbarr_to_labarr(arr):
    arr = colour.sRGB_to_XYZ(arr/255)
    arr = colour.XYZ_to_Lab(arr)
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
    vision_sort = tuple(np.lexsort(dE))
    longest_clutname_len = max([len(s) for s in clutnames])
    _fs = f'{{0:<{longest_clutname_len}}}'
    rowheaders = [_fs.format(clutnames[i]) for i in vision_sort]
    colheaders = [f'{"lowest":^8}'] + [f'{str(t):^8}' for t in combs]
    tdE = np.transpose(dE)
    ldE0 = np.min(dE, axis=0)
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


if __name__ == "__main__":

    try:
        cvd_n = int(sys.argv[1])
        start_level = int(sys.argv[2])
        n_neigh = int(sys.argv[3])
        n_clde = int(sys.argv[4])
    except (IndexError, ValueError):
        print("Usage: python generate.py <palette size> <start pyramid level> <next-level-neighbour-colors> <next-level-closest-de-colors>")
        sys.exit(1)

    np.seterr(all='raise')

    root = pathlib.Path(__file__).parent.resolve()

    img_path = root / f'cvd{cvd_n}.png'
    dE_path = root / f'cvd{cvd_n}.txt'

    haldclutdir = root.parent / "haldclut"

    cluts = (
        clut.CLUT(str(haldclutdir / "identity" / "identity.png")),       # normal
        clut.CLUT(str(haldclutdir / "cvd" / "deuta.machado2010.png")),   # deuteranopia
        clut.CLUT(str(haldclutdir / "cvd" / "prota.machado2010.png")),   # protanopia
        clut.CLUT(str(haldclutdir / "cvd" / "trita.machado2010.png")),   # tritanopia
    )

    n_vision = len(cluts)

    graypalroot = root.parent / "grayscale-palette"
    graypalp = graypalroot / f'grayscale{cvd_n:0>2}.pal'
    with graypalp.open() as f:
        graypal = tuple(rgb[0] for rgb in pal.load(f))

    graypalstrs = tuple(f'0x{v:0>2X}' for v in graypal)
    print(f'Palette gray levels: {", ".join(graypalstrs)}')

    prev_color_arrs = [None] * len(graypal)
    past_top_level = [False] * len(graypal)
    past_top_level[0] = True # black
    past_top_level[-1] = True # white

    level = start_level

    while True:

        level_img_path = root / f'cvd{cvd_n}-{level:0>2}.png'
        level_dE_path = root / f'cvd{cvd_n}-{level:0>2}.txt'


        color_arrs = [None] * len(graypal)
        isoluminantdir = root.parent / "isoluminant-depyramid"
        for i, graylvl in enumerate(graypal):
            if graylvl in {0, 255}:
                arr = np.array([[graylvl, graylvl, graylvl]], dtype="uint8")
            else:
                isoluminantimgpath = isoluminantdir / f'{graylvl:0>2x}' / f'{graylvl:0>2x}-0016-{level:0>2}.png'
                if isoluminantimgpath.is_file():
                    print(isoluminantimgpath)
                    img = Image.open(str(isoluminantimgpath))
                    if level == start_level:
                        arr = np.array(img).reshape((-1, 3))
                    else:
                        color_rgb_arr = best_pal[0][i]
                        color_lab_arr = rgbarr_to_labarr(color_rgb_arr)

                        arr0 = np.array(img).reshape((-1, 3))
                        arr0_color_i = np.where(np.all(arr0 == color_rgb_arr, axis=1))[0][0]

                        arr1 = arr0[max(0, arr0_color_i-(n_neigh//2)):min(arr0.shape[0], arr0_color_i+(n_neigh//2)+1)]

                        if n_clde:
                            # https://stackoverflow.com/a/40056251/2334951
                            # arr2 = [rgb for rgb in arr0 if rgb not in arr1]
                            A, B = arr0, arr1
                            dims = np.maximum(B.max(0),A.max(0), dtype="uint16")+1
                            out = A[~np.in1d(np.ravel_multi_index(A.T,dims),np.ravel_multi_index(B.T,dims))]
                            arr2 = out

                            arr2_lab = rgbarr_to_labarr(arr2)
                            arr2_n_colors = arr2.shape[-2]
                            color_arr2_lab = np.tile(color_lab_arr, arr2_n_colors).reshape((arr2_n_colors, 3))
                            arr2_dE = de2000.delta_e_from_lab(arr2_lab, color_arr2_lab)
                            arr3_ii = np.argsort(arr2_dE)[:n_clde]
                            arr3 = arr2[arr3_ii]

                            arr4 = np.vstack((arr1, arr3))
                            #raise Exception
                            arr = arr4
                        else:
                            arr = arr1
                else:
                    arr = prev_color_arrs[i]
                    past_top_level[i] = True
            assert arr.shape[0]
            color_arrs[i] = arr

        if all(past_top_level):
            break

        iteration_count = int(np.prod([a.shape[0] for a in color_arrs], dtype="uint64"))
        print(f'Iterations: {iteration_count}')

        idxtgen_args = [range(len(color_arrs[p])) for p in range(len(color_arrs))]
        idxtgen = itertools.product(*idxtgen_args)
        groupergen = grouper(idxtgen, batchsize)

        prevbatchnr = None
        best_pal = None
        best_dE = None
        best_sorted_dE = None

        for batchnr, batch in enumerate(groupergen, 1):
            if prevbatchnr and batchnr <= prevbatchnr:
                continue
            print(f'L{level:0>2}B{batchnr}.', end='', flush=True)
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
            sort1_dE_arr = np.sort(np.min(dE_arr, axis=1))
            sort2_dE_arr = np.sort(dE_arr.reshape((dE_arr.shape[0], dE_arr.shape[1]* dE_arr.shape[2])), axis=1)
            sort3_dE_arr = np.hstack((sort1_dE_arr, sort2_dE_arr))
            sort_ii = np.flip(np.lexsort(np.rot90(sort3_dE_arr)))
            batch_best_i = sort_ii[0]
            batch_best_dE = dE_arr[batch_best_i]
            batch_best_sorted_dE = sort3_dE_arr[batch_best_i]
            batch_best_pal = np.asarray([tuple(color_arrs[c][i]) for c, i in enumerate(b[batch_best_i])]).reshape((1, -1, 3))
            if best_sorted_dE is None or (tuple(best_sorted_dE) < tuple(batch_best_sorted_dE)):
                print("!!!", end="", flush=True)
                best_dE = batch_best_dE
                best_sorted_dE = batch_best_sorted_dE
                best_pal = batch_best_pal
                img = Image.fromarray(best_pal, 'RGB')
                img.save(level_img_path)
                print("(I)", end="", flush=True)
            with level_dE_path.open("w") as f:
                f.write(dEstr(best_dE, combs, batchnr))
            print("(B)", end="", flush=True)
            print(flush=True)
        else:
            with level_dE_path.open("w") as f:
                f.write(dEstr(best_dE, combs))
            print("E", flush=True)

        level += 1
        prev_color_arrs = color_arrs
