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

min_nci_dE = 20

if __name__ == "__main__":

    np.seterr(all='raise')

    root = pathlib.Path(__file__).parent.resolve()

    usage = """Usage: python generate.py <palette size> <pyramid level1 colors> <next-level-neighbour-colors> <next-level-closest-de-colors> [number-of-palettes (default 1)] [image of existing palettes]"""
    try:
        cvd_n = int(sys.argv[1])
        level1colors = int(sys.argv[2])
        n_neigh = int(sys.argv[3])
        n_clde = int(sys.argv[4])
    except (IndexError, ValueError):
        print(usage)
        sys.exit(1)

    img_path = root / f'cvd{cvd_n}.png'
    dE_path = root / f'cvd{cvd_n}.txt'

    if 5 < len(sys.argv):
        try:
            pal_count = int(sys.argv[5])
        except (IndexError, ValueError):
            print(usage)
            sys.exit(1)
    else:
        pal_count = 1

    if 6 < len(sys.argv):
        try:
            not_close_to_img_path = pathlib.Path(sys.argv[6])
        except (IndexError, ValueError):
            print(usage)
            sys.exit(1)
        else:
            not_close_to_img = Image.open(str(not_close_to_img_path))
            not_close_to_rgb_arr = np.array(not_close_to_img)
    else:
        not_close_to_img_path = img_path
        not_close_to_rgb_arr = np.zeros((0, cvd_n, 3))
    not_close_to_lab_arr = rgbarr_to_labarr(not_close_to_rgb_arr)

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

    for palnr in range(not_close_to_rgb_arr.shape[0] + 1, pal_count + 1):

        prev_color_arrs = [None] * len(graypal)
        past_top_level = [False] * len(graypal)
        past_top_level[0] = True # black
        past_top_level[-1] = True # white

        level = 1

        best_pal = None
        best_dE = None
        best_sorted_dE = None

        while True:

            print(f'=== LEVEL {level} ===')

            level_dE_path = root / f'cvd{cvd_n}-{level:0>2}-{palnr:0>4}.txt'
            level_img_path = root / f'cvd{cvd_n}-{level:0>2}-{palnr:0>4}.png'


            if level_dE_path.is_file() and level_img_path.is_file():
                img = Image.open(str(level_img_path))
                best_pal = np.array(img).reshape((1, -1, 3))
                best_dE, best_sorted_dE = get_pal_values(best_pal, cluts)
                del img
                with level_dE_path.open("r") as f:
                    firstline = f.readline().strip()
                if firstline.isdecimal():
                    prevbatchnr = int(firstline)
                else:
                    prevbatchnr = None
                    continue
            else:
                prevbatchnr = None

            color_arrs = [None] * len(graypal)
            isoluminantdir = root.parent / "isoluminant-depyramid"
            for i, graylvl in enumerate(graypal):
                if graylvl in {0, 255}:
                    arr = np.array([[graylvl, graylvl, graylvl]], dtype="uint8")
                else:
                    isoluminantimgpath = isoluminantdir / f'{graylvl:0>2x}' / f'{graylvl:0>2x}-{level1colors:0>4}-{level:0>2}.png'
                    if isoluminantimgpath.is_file():
                        # print(isoluminantimgpath)
                        img = Image.open(str(isoluminantimgpath))
                        if level == 1:
                            arr = np.array(img).reshape((-1, 3))
                        else:
                            color_rgb_arr = best_pal[0][i]
                            color_lab_arr = rgbarr_to_labarr(color_rgb_arr)

                            arr0 = np.array(img).reshape((-1, 3))
                            arr0_color_i = np.where(np.all(arr0 == color_rgb_arr, axis=1))[0][0]

                            exp_arr1_size = min(n_neigh, arr0.shape[0])

                            if exp_arr1_size == arr0.shape[0]:
                                arr1_i0 = arr1_i1 = 0
                                arr1_j0 = arr1_j1 = arr0.shape[0]
                                arr1 = arr0
                            else:
                                arr1_i0 = arr0_color_i-(n_neigh//2)
                                arr1_j0 = arr1_i0 + n_neigh
                                arr1_i1 = max(0, arr1_i0)
                                arr1_j1 = min(arr0.shape[0], arr1_j0)
                                arr1 = arr0[arr1_i1: arr1_j1]
                                if arr1_i0 < 0:
                                    arr1_2 = arr0[arr1_i0:]
                                    arr1 = np.vstack((arr1, arr1_2))
                                elif arr0.shape[0] < arr1_j0:
                                    arr1_2 = arr0[:arr1_j0-arr0.shape[0]]
                                    arr1 = np.vstack((arr1_2, arr1))
                            assert arr1.shape[0] == exp_arr1_size

                            exp_arr3_size = min(n_clde, arr0.shape[0] - exp_arr1_size)

                            if exp_arr3_size:
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
                                assert arr3.shape[0] == exp_arr3_size

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

            if all(past_top_level):  # exit while loop
                break

            color_lab_arrs = [rgbarr_to_labarr(a) for a in color_arrs]

            iteration_count = int(np.prod([a.shape[0] for a in color_arrs], dtype="uint64"))
            print(f'Iterations: {iteration_count} {[a.shape[0] for a in color_arrs]!r}')

            idxtgen_args = [range(len(color_arrs[p])) for p in range(len(color_arrs))]
            idxtgen = itertools.product(*idxtgen_args)
            groupergen = grouper(idxtgen, batchsize)

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
                mdE_arr = np.min(dE_arr, axis=1)
                sort1_dE_arr = np.sort(mdE_arr)
                sort2_dE_arr = np.sort(dE_arr.reshape((dE_arr.shape[0], dE_arr.shape[1]* dE_arr.shape[2])), axis=1)
                sort3_dE_arr = np.hstack((sort1_dE_arr, sort2_dE_arr))
                sort_ii = np.flip(np.lexsort(np.rot90(sort3_dE_arr)))
                for ii in sort_ii:
                    pal_rgb_arr = np.asarray([tuple(color_arrs[c][i]) for c, i in enumerate(b[ii])]).reshape((1, -1, 3))
                    pal_lab_arr = np.asarray([tuple(color_lab_arrs[c][i]) for c, i in enumerate(b[ii])]).reshape((1, -1, 3))
                    for nci in range(not_close_to_lab_arr.shape[0]):
                        nci_lab_arr = not_close_to_lab_arr[nci].reshape((1, -1, 3))
                        nci_dE_arr = de2000.delta_e_from_lab(pal_lab_arr, nci_lab_arr)[0]
                        nci_max_dE = np.max(nci_dE_arr)
                        if nci_max_dE < min_nci_dE:
                            #raise Exception
                            break
                    else:
                        break  # empty not_close_to_lab_arr
                    if min_nci_dE <= nci_max_dE:
                        #raise Exception
                        break
                else:
                    raise Exception
                    continue

                batch_best_i = ii # !!!
                batch_best_pal = np.asarray([tuple(color_arrs[c][i]) for c, i in enumerate(b[batch_best_i])]).reshape((1, -1, 3))
                batch_best_dE = dE_arr[batch_best_i]
                batch_best_sorted_dE = sort3_dE_arr[batch_best_i]
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

        not_close_to_rgb_arr = np.vstack((not_close_to_rgb_arr, best_pal))
        not_close_to_lab_arr = rgbarr_to_labarr(not_close_to_rgb_arr)
        not_close_to_img = Image.fromarray(not_close_to_rgb_arr, 'RGB')
        not_close_to_img.save(not_close_to_img_path)
