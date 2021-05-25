# 3rd party libraries
from PIL import Image
import numpy as np

# standard libraries
import collections
import pathlib
import sys

_thisdir = pathlib.Path(__file__).parent.resolve()
_parentdir = _thisdir.parent.resolve()
sys.path.insert(0, str(_parentdir))

# 3rd party libraries (module file available)
import rgbpyramid

sys.path.remove(str(_parentdir))

def rgbstr(rgb):
    r, g, b = rgb
    return f'#{r:0>2X}{g:0>2X}{b:0>2X}'


if __name__ == "__main__":

    print("rgbpyramid verification...")

    level = rgbpyramid.BOTTOMLEVEL
    while level <= rgbpyramid.TOPLEVEL:
        s = set(rgbpyramid.get_ref_rgbs(level))
        d = collections.defaultdict(lambda: 0)
        for rgb in rgbpyramid.iterrgb():
            ref_rgb = rgbpyramid.get_ref_rgb(rgb, level)
            d[ref_rgb] += 1
        assert s == set(d)
        for rgb in sorted(d, key=lambda k: (d[k], k)):
            print(f'{rgb}: {d[rgb]}')


    level = rgbpyramid.BOTTOMLEVEL

    while level < rgbpyramid.TOPLEVEL:
        print(f'== LEVEL {level}')

        for y in range(256):
            print(f'{y:0>2X}')
            img0 = Image.open(str(_thisdir / f'level{level:0>3}' / f'{y:0>2x}.png'))
            rgb0 = {tuple(a) for a in np.array(img0).reshape((-1, 3))}
            img1 = Image.open(str(_thisdir / f'level{(level*2):0>3}' / f'{y:0>2x}.png'))
            rgb1 = {tuple(a) for a in np.array(img1).reshape((-1, 3))}
            assert not rgb0 - rgb1

        level *= 2
