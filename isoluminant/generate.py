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
import de2000
import rgbpyramid

sys.path.remove(str(_parentdir))

def rgbstr(rgb):
    r, g, b = rgb
    return f'#{r:0>2X}{g:0>2X}{b:0>2X}'


if __name__ == "__main__":

    try:
        level = int(sys.argv[1])
        assert level in range(0,6)
    except (ValueError, AssertionError):
        print('Usage: python generate.py <level>')
        print('0 <= level <= 5')
        sys.exit(1)

    leveldirname = f'level{level}'
    (_thisdir / leveldirname).mkdir(parents=True, exist_ok=True)  # ensure target directory

    eh = clut.CLUT(str(_parentdir / "haldclut/gray/gray.szieberth.png"))

    print("Collecting gray levels...")

    y_rgbs = collections.defaultdict(list)
    for r, g, b in rgbpyramid.iterrgb(level):
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
        imgp = _thisdir / leveldirname / f'{y:0>2x}.png'
        img.save(str(imgp))
        shortimgp = pathlib.Path(leveldirname) / f'{y:0>2x}.png'
        print(f'"{shortimgp}" saved; colors = {len(rgbs)}')
        N += len(rgbs)
    levelvalcount = len(rgbpyramid.LEVEL[level])
    print(f'total colors = {N}; expected = {levelvalcount}^3 = {levelvalcount**3}')
