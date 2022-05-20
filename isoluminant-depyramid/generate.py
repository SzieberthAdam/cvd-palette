import itertools
import pathlib
import sys

_thisdir = pathlib.Path(__file__).parent.resolve()
_importdir = _thisdir.parent.resolve()
sys.path.insert(0, str(_importdir))

# 3rd party libraries (module file available)
import de2000

sys.path.remove(str(_importdir))

# 3rd party libraries
import colour
from PIL import Image
import numpy as np

def grouper(iterable, n):
    args = [iter(iterable)] * n
    yield from (tuple(x for x in t if x is not ...) for t in itertools.zip_longest(*args, fillvalue=...))


def rgbarr_to_labarr(arr):
    arr = colour.sRGB_to_XYZ(arr/255)
    arr = colour.XYZ_to_Lab(arr)
    return arr

def get_boundary_colors(isoluminant_level0_image_path):
    in_img_path = pathlib.Path(isoluminant_level0_image_path)
    in_img = Image.open(str(in_img_path))
    in_rgb_arr = np.array(in_img).reshape((-1, 3))
    in_lab_arr = rgbarr_to_labarr(in_rgb_arr)
    maxmax_dE = None
    maxmax_rgbs = np.array((), dtype="uint8")
    idxs_gen = grouper(itertools.combinations(range(len(in_lab_arr)), 2), 1000000)
    gen = grouper(itertools.combinations(in_lab_arr, 2), 1000000)
    for batch in gen:
        idxs = next(idxs_gen)
        lab1_arr, lab2_arr = np.array(tuple(zip(*batch)))
        delta_e_arr = de2000.delta_e_from_lab(lab1_arr, lab2_arr)
        max_dE = np.max(delta_e_arr)
        if maxmax_dE is not None and max_dE == maxmax_dE:
            max_idxs = np.where(delta_e_arr == max_dE)
            max_rgbs = np.array(
                [np.array([in_rgb_arr[j] for i in m for j in idxs[i]], dtype="uint8")
                for m in max_idxs], dtype="uint8"
            ).reshape((len(max_idxs), 2, 3))
            maxmax_rgbs = np.concatenate((maxmax_rgbs, max_rgbs))
        elif maxmax_dE is None or maxmax_dE < max_dE:
            maxmax_dE = max_dE
            max_idxs = np.where(delta_e_arr == max_dE)
            max_rgbs = np.array(
                [np.array([in_rgb_arr[j] for i in m for j in idxs[i]], dtype="uint8")
                for m in max_idxs], dtype="uint8"
            ).reshape((len(max_idxs), 2, 3))
            maxmax_rgbs = max_rgbs
    return maxmax_dE, maxmax_rgbs


if __name__ == "__main__":

    try:
        graylevel = int(sys.argv[1], 16)
        assert graylevel in range(0,256)
    except (IndexError, ValueError, AssertionError):
        print('Usage: python generate.py <graylevel>')
        print('0 <= graylevel <= FF (base 16)')
        sys.exit(1)

    root = pathlib.Path(__file__).parent.resolve()

    in_img_path = root.parent / "isoluminant" / "level0" / f'{graylevel:0>2X}.png'
    isoluminant_level0_image_path = in_img_path # devtime
    in_img_path = pathlib.Path(isoluminant_level0_image_path) # devtime
    in_img = Image.open(str(in_img_path)) # devtime
    in_rgb_arr = np.array(in_img).reshape((-1, 3)) # devtime
    in_lab_arr = rgbarr_to_labarr(in_rgb_arr) # devtime



    bound_img_path = root / f'{graylevel:0>2X}-bound.png'
    dEtxtpath = root / f'{graylevel:0>2X}-bound.de.txt'

    if bound_img_path.is_file():
        bound_img = Image.open(str(bound_img_path))
        bound_rgb_arr = np.asarray(bound_img)
        with dEtxtpath.open("r") as f:
            bound_dE = float(f.read())
    else:
        bound_dE, bound_rgb_arr = get_boundary_colors(in_img_path)
        bound_img = Image.fromarray(bound_rgb_arr, 'RGB')
        bound_img.save(str(bound_img_path))
        with dEtxtpath.open("w") as f:
            f.write(str(bound_dE))

    level01_img_path = root / f'{graylevel:0>2X}-level01.png'
    if level01_img_path.is_file():
        level01_img = Image.open(str(level01_img_path))
        level01_rgb_arr = np.asarray(level01_img)
    else:
        #bound_rgb_arr = np.asarray(bound_img)
        bound_lab_arr = rgbarr_to_labarr(bound_rgb_arr)

        color1_lab_arr = np.tile(bound_lab_arr[0][0], len(in_lab_arr)).reshape((len(in_lab_arr), 3))
        color1_dE_arr = de2000.delta_e_from_lab(in_lab_arr, color1_lab_arr)

        color2_lab_arr = np.tile(bound_lab_arr[0][1], len(in_lab_arr)).reshape((len(in_lab_arr), 3))
        color2_dE_arr = de2000.delta_e_from_lab(in_lab_arr, color2_lab_arr)

        # pairs1_lab_arr = np.hstack((in_lab_arr, color1_lab_arr)).reshape((len(in_lab_arr), 2, 3))

        a = np.vstack((color1_dE_arr, color2_dE_arr)).transpose()
        b = np.abs(np.diff(a))
        c = a[np.lexsort(np.rot90(a))]
        i = np.argmin(b)
        level01_rgb_arr = np.insert(bound_rgb_arr, 1, in_rgb_arr[i], axis=1)

        level01_img = Image.fromarray(level01_rgb_arr, 'RGB')
        level01_img.save(str(level01_img_path))

    # X = np.zeros((100000, 100000), dtype="float32")

    level02_img_path = root / f'{graylevel:0>2X}-level02.png'
    if level02_img_path.is_file():
        level02_img = Image.open(str(level02_img_path))
        level02_rgb_arr = np.asarray(level02_img)
    else:
        level01_lab_arr = rgbarr_to_labarr(level01_rgb_arr)

        # dE_arr = np.zeros((level01_lab_arr.shape[1], len(in_lab_arr)), dtype="float64")
        # for i in range(level01_lab_arr.shape[1]):
        #     color_lab_arr = np.tile(level01_lab_arr[0][i], len(in_lab_arr)).reshape((len(in_lab_arr), 3))
        #     dE_arr[i] = color_dE_arr = de2000.delta_e_from_lab(in_lab_arr, color_lab_arr)

        dE_arr = np.zeros((len(in_lab_arr), level01_lab_arr.shape[1]), dtype="float64")
        for i in range(level01_lab_arr.shape[1]):
            color_lab_arr = np.tile(level01_lab_arr[0][i], len(in_lab_arr)).reshape((len(in_lab_arr), 3))
            dE_arr[:, i] = color_dE_arr = de2000.delta_e_from_lab(in_lab_arr, color_lab_arr)
