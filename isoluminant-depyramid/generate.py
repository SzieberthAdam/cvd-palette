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
    rgb_arr = np.array(in_img).reshape((-1, 3))
    lab_arr = rgbarr_to_labarr(rgb_arr)
    maxmax_dE = None
    maxmax_rgbs = np.array((), dtype="uint8")
    idxs_gen = grouper(itertools.combinations(range(len(lab_arr)), 2), 1000000)
    gen = grouper(itertools.combinations(lab_arr, 2), 1000000)
    for batch in gen:
        idxs = next(idxs_gen)
        lab1_arr, lab2_arr = np.array(tuple(zip(*batch)))
        delta_e_arr = de2000.delta_e_from_lab(lab1_arr, lab2_arr)
        max_dE = np.max(delta_e_arr)
        if maxmax_dE is not None and max_dE == maxmax_dE:
            max_idxs = np.where(delta_e_arr == max_dE)
            max_rgbs = np.array(
                [np.array([rgb_arr[j] for i in m for j in idxs[i]], dtype="uint8")
                for m in max_idxs], dtype="uint8"
            ).reshape((len(max_idxs), 2, 3))
            maxmax_rgbs = np.concatenate((maxmax_rgbs, max_rgbs))
        elif maxmax_dE is None or maxmax_dE < max_dE:
            maxmax_dE = max_dE
            max_idxs = np.where(delta_e_arr == max_dE)
            max_rgbs = np.array(
                [np.array([rgb_arr[j] for i in m for j in idxs[i]], dtype="uint8")
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

    max_dE, max_rgb_arr = get_boundary_colors(in_img_path)

    out_img_path = root / f'bound-{graylevel:0>2X}.png'
    out_img = Image.fromarray(max_rgb_arr, 'RGB')
    out_img.save(str(out_img_path))
    dEtxtpath = root / f'bound-{graylevel:0>2X}.de.txt'
    with dEtxtpath.open("w") as f:
        f.write(str(max_dE))
