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


def get_dE_arr(rgb_arr):
    haldclutdir = _thisdir.parent / "haldclut"
    cluts = (
        clut.CLUT(str(haldclutdir / "identity" / "identity.png")),       # normal
        clut.CLUT(str(haldclutdir / "cvd" / "deuta.machado2010.png")),   # deuteranopia
        clut.CLUT(str(haldclutdir / "cvd" / "prota.machado2010.png")),   # protanopia
        clut.CLUT(str(haldclutdir / "cvd" / "trita.machado2010.png")),   # tritanopia
    )
    n_vision = len(cluts)
    n_pals = rgb_arr.shape[0]
    n_colors = rgb_arr.shape[-2]
    combs = tuple(itertools.combinations(range(n_colors), 2))
    n_pairs = len(combs)
    dE_arr = np.zeros((n_pals, n_pairs, n_vision), dtype="float64")
    for cti, clut_ in enumerate(cluts):
        clut_rgb_arr = clut_(rgb_arr)
        clut_lab_arr = rgbarr_to_labarr(clut_rgb_arr)
        for coi, (c1i, c2i) in enumerate(combs):
            lab1 = clut_lab_arr[:, c1i, :]
            lab2 = clut_lab_arr[:, c2i, :]
            this_dE_arr = de2000.delta_e_from_lab(lab1, lab2)
            dE_arr[:,coi][:,cti] = this_dE_arr
    return dE_arr


def argsort_rgb_arr_keys(rgb_arr, *, dE_arr=None):
    dE_arr = (get_dE_arr(rgb_arr) if dE_arr is None else dE_arr)
    min_dE_by_vision = np.min(dE_arr, axis=1)  # shape = (n_pals, n_vision)
    argsort_dE_arr_by_vision = np.argsort(min_dE_by_vision)  # shape = (n_pals, n_vision)
    #sort_dE_arr_by_vision = np.sort(min_dE_by_vision)  # shape = (n_pals, n_vision)
    #True: np.all(np.take_along_axis(min_dE_by_vision, argsort_dE_arr_by_vision, 1) == np.sort(min_dE_by_vision))
    sort_dE_arr_by_vision = np.take_along_axis(min_dE_by_vision, argsort_dE_arr_by_vision, 1)  # shape = (n_pals, n_vision)
    min_dE_by_pairs = np.min(dE_arr, axis=2) # shape = (n_pals, n_pairs)
    argsort_dE_arr_by_pairs = np.argsort(min_dE_by_pairs)  # shape = (n_pals, n_pairs)
    #sort_dE_arr_by_pairs = np.sort(min_dE_by_pairs)  # shape = (n_pals, n_pairs)
    #True: np.all(np.take_along_axis(min_dE_by_pairs, argsort_dE_arr_by_pairs, 1) == np.sort(min_dE_by_pairs))
    sort_dE_arr_by_pairs = np.take_along_axis(min_dE_by_pairs, argsort_dE_arr_by_pairs, 1)  # shape = (n_pals, n_pairs)
    sort_dE_arr_mixed = np.sort(dE_arr.reshape((dE_arr.shape[0], dE_arr.shape[1]* dE_arr.shape[2])), axis=1)  # shape = (n_pals, n_vision * n_pairs)
    sort_dE_arr = np.hstack((sort_dE_arr_by_vision, sort_dE_arr_by_pairs, sort_dE_arr_mixed))  # shape = (n_pals, n_vision + n_pairs + n_vision * n_pairs)
    return sort_dE_arr


def argsort_rgb_arr(rgb_arr, *, dE_arr=None):
    sort_dE_arr = argsort_rgb_arr_keys(rgb_arr, dE_arr=dE_arr)
    sort_ii = np.flip(np.lexsort(np.rot90(sort_dE_arr)))  # shape = (n_pals,)
    return sort_ii


def sort_rgb_arr(rgb_arr, *, dE_arr=None):
    sort_ii = argsort_rgb_arr(rgb_arr, dE_arr=dE_arr)
    sorted_rgb_arr = rgb_arr[sort_ii]
    return sorted_rgb_arr


def report_str(rgb_arr, *, dE_arr=None):
    haldclutdir = _thisdir.parent / "haldclut"
    cluts = (
        clut.CLUT(str(haldclutdir / "identity" / "identity.png")),       # normal
        clut.CLUT(str(haldclutdir / "cvd" / "deuta.machado2010.png")),   # deuteranopia
        clut.CLUT(str(haldclutdir / "cvd" / "prota.machado2010.png")),   # protanopia
        clut.CLUT(str(haldclutdir / "cvd" / "trita.machado2010.png")),   # tritanopia
    )
    dE_arr = dE_arr or get_dE_arr(rgb_arr)

    n_pals = rgb_arr.shape[0]
    n_colors = rgb_arr.shape[-2]
    combs = tuple(itertools.combinations(range(n_colors), 2))

    clut_rgb_arrs = {clutnames[cti]: clut_(rgb_arr) for cti, clut_ in enumerate(cluts)}

    rowheaders = np.array([f'C{c1i+1} vs. C{c2i+1}' for c1i, c2i in combs])
    rowheaders_len = max((len(s) for s in rowheaders))
    rowheader_fs = f'{{:<{rowheaders_len}}}'
    rowheader_sep = " | "
    colheaders = np.array(clutnames)
    colheaders_len = max(12, max((len(s) for s in colheaders)))
    colheader_fs = f'{{:^{colheaders_len}}}'
    colheader_sep = "   "
    val_fs = f'{{:>{colheaders_len}.{colheaders_len-2-4}f}}'
    val_sep = "   "
    lowestpairval_sep = "   "
    lowestpairval_fs = f'{{:>{colheaders_len-2}.{colheaders_len-2-4}f}}'
    lowestheader_fs = f'{{:>{colheaders_len-2}}}'
    lowestvisionval_sep = " --"
    pairsheader_fs = f'{{:^{rowheaders_len}}}'
    rgbstr_fs = f'{{:^{colheaders_len}}}'
    palnr_fs = f'{{:^{rowheaders_len}}}'

    pal_strs = []

    for pi in range(n_pals):
        min_dE_by_vision = np.min(dE_arr[pi], axis=0)  # shape = (n_pals, n_vision)
        vii = np.argsort(min_dE_by_vision)
        min_dE_by_pairs = np.min(dE_arr[pi], axis=1) # shape = (n_pals, n_pairs)
        pai = np.argsort(min_dE_by_pairs)

        pal_dE_arr_2dsorted = dE_arr[pi][pai][:,vii]
        rowheaders_2dsorted = np.array(rowheaders)[pai]
        colheaders_2dsorted = np.array(colheaders)[vii]
        pal_dE_arr_2dsorted_strs = np.zeros(pal_dE_arr_2dsorted.shape, dtype=f'<U{max(rowheaders_len, colheaders_len)}')
        for ri, row in enumerate(pal_dE_arr_2dsorted):
            srow = []
            for vi, val in enumerate(row):
                s = val_fs.format(val)
                ismin_by_vision = (val == min_dE_by_vision[vii][vi])
                ismin_by_pairs = (val == min_dE_by_pairs[pai][ri])
                if ismin_by_vision and ismin_by_pairs:
                    s = "#" + s[1:]
                elif ismin_by_vision:
                    s = '§' + s[1:]
                elif ismin_by_pairs:
                    s = '*' + s[1:]
                pal_dE_arr_2dsorted_strs[ri][vi] = s
        colheader_strs = [colheader_fs.format(s) for s in colheaders_2dsorted]
        rowheader_strs = [rowheader_fs.format(s) for s in rowheaders_2dsorted]
        colheader_str = colheader_sep.join(colheader_strs)

        lines2 = []
        s = ""
        s += " " * rowheaders_len
        s += " " * len(rowheader_sep)
        s += " " * (colheaders_len -2)
        s += colheader_sep
        s += colheader_str
        lines2.append(s)
        s = ""
        s += " " * rowheaders_len
        s += " " * len(rowheader_sep)
        s += " " * (colheaders_len -2)
        s += colheader_sep
        s += colheader_sep.join(["="*colheaders_len for _ in colheaders])
        lines2.append(s)
        s = ""
        s += pairsheader_fs.format("pairs")
        s += " " * len(rowheader_sep)
        s += lowestheader_fs.format("lowest")
        s += lowestvisionval_sep
        s += lowestvisionval_sep.join([val_fs.format(val) for val in min_dE_by_vision[vii]])
        lines2.append(s)
        s = ""
        s += "=" * rowheaders_len
        s += " " * len(rowheader_sep)
        s += lowestheader_fs.format("    | ")
        s += colheader_sep
        s += colheader_sep.join(["-" * colheaders_len for _ in colheaders])
        lines2.append(s)
        for ri, row in enumerate(pal_dE_arr_2dsorted_strs):
            s = ""
            s += rowheader_strs[ri]
            s += rowheader_sep
            s += lowestpairval_fs.format(min_dE_by_pairs[pai][ri])
            s += lowestpairval_sep
            s += val_sep.join(row)
            lines2.append(s)

        s_table2 = "\n".join(lines2)

        lines1 = []
        s = ""
        s += palnr_fs.format(f'[ {pi+1} ]')
        s += " " * len(rowheader_sep)
        s += " " * (colheaders_len -2 -5)
        s += "color"
        s += colheader_sep
        s += colheader_str.replace("  normal  ", "* normal *")
        lines1.append(s)
        s = ""
        s += " " * rowheaders_len
        s += " " * len(rowheader_sep)
        s += " " * (colheaders_len -2 -5)
        s += "====="
        s += colheader_sep
        s += colheader_sep.join(["="*colheaders_len for _ in colheaders])
        lines1.append(s)
        for ci in range(n_colors):
            s = ""
            s += " " * rowheaders_len
            s += " " * len(rowheader_sep)
            s += " " * (colheaders_len -2 -5)
            s += f'{"C" + str(ci+1):^5}'
            s += colheader_sep
            rgbstrs = []
            for vi in vii:
                clutname = clutnames[vi]
                rgb = clut_rgb_arrs[clutname][pi][ci]
                rgbstr_ = rgbstr(rgb)
                rgbstrs.append(rgbstr_fs.format(rgbstr_))
            s += val_sep.join(rgbstrs)
            lines1.append(s)

        s_table1 = "\n".join(lines1)

        s_pal_all = "\n".join([s_table1, "", s_table2, ""])

        pal_strs.append(s_pal_all)

    s_all = f'\n{"#"*len(lines2[0])}\n\n'.join(pal_strs)
    return s_all


if __name__ == "__main__":

    np.seterr(all='raise')

    root = pathlib.Path(__file__).parent.resolve()

    usage = """Usage: python sortpals.py <image of palettes>"""

    try:
        img_path = pathlib.Path(sys.argv[1])
    except (IndexError, ValueError, FileNotFoundError):
        print(usage)
        sys.exit(1)

    img = Image.open(str(img_path))
    rgb_arr = np.array(img)

    haldclutdir = root.parent / "haldclut"

    cluts = (
        clut.CLUT(str(haldclutdir / "identity" / "identity.png")),       # normal
        clut.CLUT(str(haldclutdir / "cvd" / "deuta.machado2010.png")),   # deuteranopia
        clut.CLUT(str(haldclutdir / "cvd" / "prota.machado2010.png")),   # protanopia
        clut.CLUT(str(haldclutdir / "cvd" / "trita.machado2010.png")),   # tritanopia
    )

    n_vision = len(cluts)
    n_pals = rgb_arr.shape[0]
    n_colors = rgb_arr.shape[-2]
    combs = tuple(itertools.combinations(range(n_colors), 2))
    n_pairs = len(combs)
    dE_arr = np.zeros((n_pals, n_pairs, n_vision), dtype="float64")
    for cti, clut_ in enumerate(cluts):
        clut_rgb_arr = clut_(rgb_arr)
        clut_lab_arr = rgbarr_to_labarr(clut_rgb_arr)
        for coi, (c1i, c2i) in enumerate(combs):
            lab1 = clut_lab_arr[:, c1i, :]
            lab2 = clut_lab_arr[:, c2i, :]
            this_dE_arr = de2000.delta_e_from_lab(lab1, lab2)
            dE_arr[:,coi][:,cti] = this_dE_arr

    min_dE_by_vision = np.min(dE_arr, axis=1)  # shape = (n_pals, n_vision)
    argsort_dE_arr_by_vision = np.argsort(min_dE_by_vision)  # shape = (n_pals, n_vision)
    #sort_dE_arr_by_vision = np.sort(min_dE_by_vision)  # shape = (n_pals, n_vision)
    #True: np.all(np.take_along_axis(min_dE_by_vision, argsort_dE_arr_by_vision, 1) == np.sort(min_dE_by_vision))
    sort_dE_arr_by_vision = np.take_along_axis(min_dE_by_vision, argsort_dE_arr_by_vision, 1)  # shape = (n_pals, n_vision)
    min_dE_by_pairs = np.min(dE_arr, axis=2) # shape = (n_pals, n_pairs)
    argsort_dE_arr_by_pairs = np.argsort(min_dE_by_pairs)  # shape = (n_pals, n_pairs)
    #sort_dE_arr_by_pairs = np.sort(min_dE_by_pairs)  # shape = (n_pals, n_pairs)
    #True: np.all(np.take_along_axis(min_dE_by_pairs, argsort_dE_arr_by_pairs, 1) == np.sort(min_dE_by_pairs))
    sort_dE_arr_by_pairs = np.take_along_axis(min_dE_by_pairs, argsort_dE_arr_by_pairs, 1)  # shape = (n_pals, n_pairs)
    sort_dE_arr_mixed = np.sort(dE_arr.reshape((dE_arr.shape[0], dE_arr.shape[1]* dE_arr.shape[2])), axis=1)  # shape = (n_pals, n_vision * n_pairs)
    sort_dE_arr = np.hstack((sort_dE_arr_by_vision, sort_dE_arr_by_pairs, sort_dE_arr_mixed))  # shape = (n_pals, n_vision + n_pairs + n_vision * n_pairs)
    sort_ii = np.flip(np.lexsort(np.rot90(sort_dE_arr)))  # shape = (n_pals,)
    sorted_rgb_arr = rgb_arr[sort_ii]
    sorted_dE_arr = sort_dE_arr[sort_ii]

    sorted_img_path = img_path.with_suffix(".sorted" + img_path.suffix)
    sorted_img = Image.fromarray(sorted_rgb_arr, 'RGB')
    sorted_img.save(sorted_img_path)

    # replace original arrays with sorted ones
    rgb_arr = rgb_arr[sort_ii]
    dE_arr = dE_arr[sort_ii]

    clut_rgb_arrs = {clutnames[cti]: clut_(rgb_arr) for cti, clut_ in enumerate(cluts)}

    rowheaders = np.array([f'C{c1i+1} vs. C{c2i+1}' for c1i, c2i in combs])
    rowheaders_len = max((len(s) for s in rowheaders))
    rowheader_fs = f'{{:<{rowheaders_len}}}'
    rowheader_sep = " | "
    colheaders = np.array(clutnames)
    colheaders_len = max(12, max((len(s) for s in colheaders)))
    colheader_fs = f'{{:^{colheaders_len}}}'
    colheader_sep = "   "
    val_fs = f'{{:>{colheaders_len}.{colheaders_len-2-4}f}}'
    val_sep = "   "
    lowestpairval_sep = "   "
    lowestpairval_fs = f'{{:>{colheaders_len-2}.{colheaders_len-2-4}f}}'
    lowestheader_fs = f'{{:>{colheaders_len-2}}}'
    lowestvisionval_sep = " --"
    pairsheader_fs = f'{{:^{rowheaders_len}}}'
    rgbstr_fs = f'{{:^{colheaders_len}}}'
    palnr_fs = f'{{:^{rowheaders_len}}}'

    pal_strs = []

    for pi in range(n_pals):
        min_dE_by_vision = np.min(dE_arr[pi], axis=0)  # shape = (n_pals, n_vision)
        vii = np.argsort(min_dE_by_vision)
        min_dE_by_pairs = np.min(dE_arr[pi], axis=1) # shape = (n_pals, n_pairs)
        pai = np.argsort(min_dE_by_pairs)

        pal_dE_arr_2dsorted = dE_arr[pi][pai][:,vii]
        rowheaders_2dsorted = np.array(rowheaders)[pai]
        colheaders_2dsorted = np.array(colheaders)[vii]
        pal_dE_arr_2dsorted_strs = np.zeros(pal_dE_arr_2dsorted.shape, dtype=f'<U{max(rowheaders_len, colheaders_len)}')
        for ri, row in enumerate(pal_dE_arr_2dsorted):
            srow = []
            for vi, val in enumerate(row):
                s = val_fs.format(val)
                ismin_by_vision = (val == min_dE_by_vision[vii][vi])
                ismin_by_pairs = (val == min_dE_by_pairs[pai][ri])
                if ismin_by_vision and ismin_by_pairs:
                    s = "#" + s[1:]
                elif ismin_by_vision:
                    s = '§' + s[1:]
                elif ismin_by_pairs:
                    s = '*' + s[1:]
                pal_dE_arr_2dsorted_strs[ri][vi] = s
        colheader_strs = [colheader_fs.format(s) for s in colheaders_2dsorted]
        rowheader_strs = [rowheader_fs.format(s) for s in rowheaders_2dsorted]
        colheader_str = colheader_sep.join(colheader_strs)

        lines2 = []
        s = ""
        s += " " * rowheaders_len
        s += " " * len(rowheader_sep)
        s += " " * (colheaders_len -2)
        s += colheader_sep
        s += colheader_str
        lines2.append(s)
        s = ""
        s += " " * rowheaders_len
        s += " " * len(rowheader_sep)
        s += " " * (colheaders_len -2)
        s += colheader_sep
        s += colheader_sep.join(["="*colheaders_len for _ in colheaders])
        lines2.append(s)
        s = ""
        s += pairsheader_fs.format("pairs")
        s += " " * len(rowheader_sep)
        s += lowestheader_fs.format("lowest")
        s += lowestvisionval_sep
        s += lowestvisionval_sep.join([val_fs.format(val) for val in min_dE_by_vision[vii]])
        lines2.append(s)
        s = ""
        s += "=" * rowheaders_len
        s += " " * len(rowheader_sep)
        s += lowestheader_fs.format("    | ")
        s += colheader_sep
        s += colheader_sep.join(["-" * colheaders_len for _ in colheaders])
        lines2.append(s)
        for ri, row in enumerate(pal_dE_arr_2dsorted_strs):
            s = ""
            s += rowheader_strs[ri]
            s += rowheader_sep
            s += lowestpairval_fs.format(min_dE_by_pairs[pai][ri])
            s += lowestpairval_sep
            s += val_sep.join(row)
            lines2.append(s)

        s_table2 = "\n".join(lines2)

        lines1 = []
        s = ""
        s += palnr_fs.format(f'[ {pi+1} ]')
        s += " " * len(rowheader_sep)
        s += " " * (colheaders_len -2 -5)
        s += "color"
        s += colheader_sep
        s += colheader_str.replace("  normal  ", "* normal *")
        lines1.append(s)
        s = ""
        s += " " * rowheaders_len
        s += " " * len(rowheader_sep)
        s += " " * (colheaders_len -2 -5)
        s += "====="
        s += colheader_sep
        s += colheader_sep.join(["="*colheaders_len for _ in colheaders])
        lines1.append(s)
        for ci in range(n_colors):
            s = ""
            s += " " * rowheaders_len
            s += " " * len(rowheader_sep)
            s += " " * (colheaders_len -2 -5)
            s += f'{"C" + str(ci+1):^5}'
            s += colheader_sep
            rgbstrs = []
            for vi in vii:
                clutname = clutnames[vi]
                rgb = clut_rgb_arrs[clutname][pi][ci]
                rgbstr_ = rgbstr(rgb)
                rgbstrs.append(rgbstr_fs.format(rgbstr_))
            s += val_sep.join(rgbstrs)
            lines1.append(s)

        s_table1 = "\n".join(lines1)

        s_pal_all = "\n".join([s_table1, "", s_table2, ""])

        pal_strs.append(s_pal_all)

    s_all = f'\n{"#"*len(lines2[0])}\n\n'.join(pal_strs)

    sorted_dEtxt_path = img_path.with_suffix(".sorted" + ".txt")
    with sorted_dEtxt_path.open("w", encoding="utf8", newline='\r\n') as f:
        f.write(s_all)
