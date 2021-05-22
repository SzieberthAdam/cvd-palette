import colorsys
import os
import pathlib
import sys

import numpy as np
from PIL import Image

workdir = pathlib.Path(os.getcwd()).resolve()


def sort_by_hsv(img):
    arr = np.array(img)
    rgb_arr = arr.reshape(-1, 3)
    hsv_arr = np.apply_along_axis(lambda rgb: colorsys.rgb_to_hsv(*rgb), -1, rgb_arr.astype("float64"))
    sorted_idxs = np.lexsort(np.rot90(hsv_arr))
    new_arr = rgb_arr[sorted_idxs].reshape(arr.shape)
    new_img = Image.fromarray(new_arr, 'RGB')
    return new_img


if __name__ == "__main__":

    argvidx = 1
    if argvidx < len(sys.argv) and sys.argv[argvidx].lower() == "overwrite":
        overwrite = True
        argvidx += 1
    else:
        overwrite= False

    filename = (sys.argv[argvidx] if argvidx < len(sys.argv) else "*.png")
    try:
        p = pathlib.Path(str(filename)).resolve()
    except OSError:
        files = workdir.glob(filename)
    else:
        files = [p]

    for p in files:
        img = Image.open(str(p))
        new_img = sort_by_hsv(img)
        if overwrite:
            new_p = p
        else:
            new_p = p.parent / f'{p.stem}.byhsv{p.suffix}'
        new_img.save(str(new_p))

    sys.exit(0)
