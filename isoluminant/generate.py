# 3rd party libraries
from PIL import Image
import numpy as np

# standard libraries
import collections
import itertools
import pathlib
import statistics
import sys

_thisdir = pathlib.Path(__file__).parent.resolve()
_parentdir = _thisdir.parent.resolve()
sys.path.insert(0, str(_parentdir))

# 3rd party libraries (module file available)
import clut

sys.path.remove(str(_parentdir))

def iterrgb():
    for r in range(256):
        for g in range(256):
            for b in range(256):
                yield r, g, b


def rgbstr(rgb):
    r, g, b = rgb
    return f'#{r:0>2X}{g:0>2X}{b:0>2X}'


if __name__ == "__main__":

    if 1 < len(sys.argv):
        clutname = sys.argv[1]
    else:
        clutname = "szieberth"

    (_thisdir / clutname).mkdir(parents=True, exist_ok=True)  # ensure target directory

    eh = clut.CLUT(str(_parentdir / f'haldclut/gray/gray.{clutname}.png'))

    print("Collecting gray levels...")

    y_rgbs = collections.defaultdict(list)
    for r, g, b in iterrgb():
        y = eh.clut[r][g][b][0]
        y_rgbs[y].append((r, g, b))

    print("Done.")

    y_rgbs_arr = {}
    N = 0
    for y, rgbs in y_rgbs.items():
        arr = np.array(rgbs, dtype=np.uint8)  # 0..255
        y_rgbs_arr[y] = arr
        shape = [1] + list(arr.shape)
        arr = arr.reshape(shape)
        img = Image.fromarray(arr, mode='RGB')
        imgp = _thisdir / clutname / f'{y:0>2x}.png'
        img.save(str(imgp))
        shortimgp = pathlib.Path(clutname) / f'{y:0>2x}.png'
        print(f'"{shortimgp}" saved; colors = {len(rgbs)}')
        N += len(rgbs)
    print(f'total colors = {N}; expected = 256^3 = {256**3}')
