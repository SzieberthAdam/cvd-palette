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

    if len(sys.argv) < 2:
        level = rgbpyramid.TOPLEVEL
    else:
        try:
            level = int(sys.argv[1])
            assert 1 < level
        except (ValueError, AssertionError):
            print(f'ERROR! 1< integer palette size was expected as argument.')
            print(f'Hint: {rgbpyramid.TOPLEVEL} represents full palette.')
            print(f'Hint: Otherwise, a power of two plus one value (3, 5, 9, 17, 33, ...) is recommended.')
            sys.exit(1)

    if 2 < len(sys.argv):
        try:
            ys = [int(sys.argv[2])]
        except (ValueError, AssertionError):
            print(f'ERROR! integer grayvalue was expected as optional second argument.')
            sys.exit(2)
    else:
        ys = tuple(range(256))


    leveldirname = f'level{level:0>3}'
    (_thisdir / leveldirname).mkdir(parents=True, exist_ok=True)  # ensure target directory

    eh = clut.CLUT(str(_parentdir / "haldclut/gray/gray.szieberth.png"))

    if level == rgbpyramid.TOPLEVEL:

        y_rgbs = collections.defaultdict(list)
        for r, g, b in rgbpyramid.iterrgb():
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
            imgp = _thisdir / leveldirname / f'{y:0>2x}.png'
            img.save(str(imgp))
            y_rgbs_img[y] = img
            N += len(rgbs)
        print("-"*30)
        print(f'N={N}; expected=256^3={256**3}')

    else:

        _NORGB = (None, None)
        y_rgbs = collections.defaultdict(dict)

        for rgb in rgbpyramid.iterrgb():
            r, g, b = rgb
            if g == 255 and b == 255:
                print(rgbstr(rgb))
            y = eh.clut[r][g][b][0]
            d = y_rgbs[y]
            refrgb = rgbpyramid.get_ref_rgb(rgb, level)
            currdistance, currrgb = d.get(refrgb, _NORGB)
            distance = rgbpyramid.get_distance(rgb, refrgb)
            if currdistance is None or distance < currdistance:
                d[refrgb] = (distance, rgb)
            elif distance == currdistance:
                currdE = de2000.get_pal_delta_e_pairs([currrgb, refrgb])
                dE = de2000.get_pal_delta_e_pairs([rgb, refrgb])
                if dE < currdE:
                    d[refrgb] = (distance, rgb)

        for y, d in y_rgbs.items():
            rgbs = sorted([rgb for distance, rgb in d.values()])

            imgout = Image.new('RGB', [len(rgbs), 1])
            imgout.putdata(rgbs)
            imgoutp = _thisdir / leveldirname / f'{y:0>2x}.png'
            imgout.save(imgoutp)
            print(f'"{y:0>2x}.png" saved.')
