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
from PIL import Image
import numpy as np


import sortall


if __name__ == "__main__":

    usage = """Usage: python splitall.py <palette image>"""
    try:
        in_img_path = pathlib.Path(sys.argv[1])
    except (IndexError, ValueError):
        print(usage)
        sys.exit(1)

    in_img = Image.open(str(in_img_path))
    in_rgb_arr = np.array(in_img, dtype="uint8")
    for n, row_rgb_arr in enumerate(in_rgb_arr, 1):
        out_rgb_arr = row_rgb_arr.reshape((1, -1, 3))
        out_img_path = in_img_path.parent / pathlib.Path(f'{in_img_path.stem}-{n:0>4}{in_img_path.suffix}')
        out_img = Image.fromarray(out_rgb_arr, 'RGB')
        out_img.save(out_img_path)

        report_str = sortall.report_str(out_rgb_arr)
        report_str_path = out_img_path.with_suffix(".txt")
        with report_str_path.open("w", encoding="utf8", newline='\r\n') as f:
            f.write(report_str)
