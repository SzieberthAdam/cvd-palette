# 3rd party libraries
from PIL import Image
import numpy as np

# 3rd party libraries (module file available)
import clut

# standard libraries
import collections
import pathlib
import sys


def iterrgb():
    for r in range(256):
        for g in range(256):
            for b in range(256):
                yield r, g, b


if __name__ == "__main__":

    if 2 <= len(sys.argv):
        subdirname = sys.argv[1]
    else:
        subdirname = "monocolors"

    root = pathlib.Path(__file__).parent
    directory = root / subdirname
    directory.mkdir(parents=True, exist_ok=True)  # ensure directory
    eh = clut.CLUT(str(root / "haldclut/cvd.mono.achromatopsia.png"))

    y_rgbs = collections.defaultdict(list)
    for r, g, b in iterrgb():
        y = eh.clut[r][g][b][0]
        y_rgbs[y].append((r, g, b))

    y_rgbs_arr = {}
    y_rgbs_img = {}
    N = 0
    for y, rgbs in y_rgbs.items():
        print(f'Gray 0x{y:0>2X}; n={len(rgbs)}')
        arr = np.array(rgbs, dtype=np.uint8)  # 0..255
        y_rgbs_arr[y] = arr
        shape = [1] + list(arr.shape)
        arr = arr.reshape(shape)
        img = Image.fromarray(arr, mode='RGB')
        img.save(f'{directory}/{y:0>2x}.png')
        y_rgbs_img[y] = img
        N += len(rgbs)
    print("-"*30)
    print(f'N={N}; expected=256^3={256**3}')
