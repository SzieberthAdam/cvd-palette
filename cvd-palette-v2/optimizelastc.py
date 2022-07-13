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


import optimizelast1
import optimizelast2
import sortall

def main(in_rgb_arr, n_neighbour_colors, n_close_dE_colors, min_nci_dE=10.0):

    optimized = False
    optimized1, optimized2 = True, True
    out_rgb_arr = in_rgb_arr
    while optimized1 or optimized2:
        optimized2, out_rgb_arr = optimizelast2.main(out_rgb_arr, n_neighbour_colors, n_close_dE_colors, min_nci_dE=min_nci_dE)
        optimized1, out_rgb_arr = optimizelast1.main(out_rgb_arr, min_nci_dE=min_nci_dE)
        optimized = optimized1 or optimized2
    return optimized, out_rgb_arr


if __name__ == "__main__":
    usage = """Usage: python optimizelastc.py <palette image> <next-level-neighbour-colors> <next-level-closest-de-colors> [min difference]"""
    try:
        in_img_path = pathlib.Path(sys.argv[1])
    except (IndexError, ValueError):
        print(usage)
        sys.exit(1)

    try:
        n_neighbour_colors = int(sys.argv[2])
        n_close_dE_colors = int(sys.argv[3])
    except (IndexError, ValueError):
        print(usage)
        sys.exit(1)

    if 4 < len(sys.argv):
        try:
            min_nci_dE = float(sys.argv[4])
        except (ValueError):
            print(usage)
            sys.exit(2)
    else:
        min_nci_dE = 10.0

    in_img = Image.open(str(in_img_path))
    in_rgb_arr = np.array(in_img, dtype="uint8")

    optimized, out_rgb_arr = main(in_rgb_arr, n_neighbour_colors, n_close_dE_colors, min_nci_dE)

    out_img_path = in_img_path.with_suffix(f'.optc' + in_img_path.suffix)
    out_dE_path = out_img_path.with_suffix(".txt")

    out_img = Image.fromarray(out_rgb_arr, 'RGB')
    out_img.save(out_img_path)
    report_str = sortall.report_str(out_rgb_arr)
    with out_dE_path.open("w", encoding="utf8", newline='\r\n') as f:
        f.write(report_str)
