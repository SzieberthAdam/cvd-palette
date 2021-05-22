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

sys.path.remove(str(_parentdir))

TOPLEVEL = 256

def iterrgb(points=None):
    points = points or tuple(range(256))
    for r in points:
        for g in points:
            for b in points:
                yield r, g, b

def get_ref_rgb(rgb, dist):
    result = [None] * 3
    for i in range(3):
        d, m = divmod(rgb[i], dist)
        v = dist * (d + (1 if dist <= 2*m else 0))
        result[i] = v
    return tuple(result)

def get_distance(rgb1, rgb2):
    return (rgb1[0]-rgb2[0])**2 + (rgb1[1]-rgb2[1])**2 + (rgb1[2]-rgb2[2])**2

def quantiles(data, *, n=4, method='exclusive'):
    """Divide *data* into *n* continuous intervals with equal probability.

    Returns a list of (n - 1) cut points separating the intervals.

    Set *n* to 4 for quartiles (the default).  Set *n* to 10 for deciles.
    Set *n* to 100 for percentiles which gives the 99 cuts points that
    separate *data* in to 100 equal sized groups.

    The *data* can be any iterable containing sample.
    The cut points are linearly interpolated between data points.

    If *method* is set to *inclusive*, *data* is treated as population
    data.  The minimum value is treated as the 0th percentile and the
    maximum value is treated as the 100th percentile.
    """
    if n < 1:
        raise StatisticsError('n must be at least 1')
    data = sorted(data)
    ld = len(data)
    if ld < 2:
        raise StatisticsError('must have at least two data points')
    if method == 'inclusive':
        print("here")
        m = ld - 1
        result = []
        for i in range(1, n):
            j = i * m // n
            delta = i*m - j*n
            interpolated = (data[j] * (n - delta) + data[j+1] * delta) / n
            result.append(interpolated)
        return result
    if method == 'exclusive':
        m = ld + 1
        result = []
        for i in range(1, n):
            j = i * m // n                               # rescale i to m/n
            j = 1 if j < 1 else ld-1 if j > ld-1 else j  # clamp to 1 .. ld-1
            delta = i*m - j*n                            # exact integer math
            interpolated = (data[j-1] * (n - delta) + data[j] * delta) / n
            result.append(interpolated)
        return result
    raise ValueError(f'Unknown method: {method!r}')

def rgbstr(rgb):
    r, g, b = rgb
    return f'#{r:0>2X}{g:0>2X}{b:0>2X}'


if __name__ == "__main__":

    if len(sys.argv) < 2:
        level = TOPLEVEL
    else:
        try:
            level = int(sys.argv[1])
            assert 1 < level
        except (ValueError, AssertionError):
            print(f'ERROR! 1< integer palette size was expected as argument.')
            print(f'Hint: {TOPLEVEL} represents full palette.')
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

    if level == TOPLEVEL:

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
            imgp = _thisdir / leveldirname / f'{y:0>2x}.png'
            img.save(str(imgp))
            y_rgbs_img[y] = img
            N += len(rgbs)
        print("-"*30)
        print(f'N={N}; expected=256^3={256**3}')

    else:

        _NORGB = (None, None)
        dist = int(256 / level)

        refpoints = list(range(0, 257, dist))
        y_rgbs = collections.defaultdict(dict)

        for rgb in iterrgb():
            r, g, b = rgb
            if g == 255 and b == 255:
                print(rgbstr(rgb))
            y = eh.clut[r][g][b][0]
            d = y_rgbs[y]
            refrgb = get_ref_rgb(rgb, dist)
            currdistance, currrgb = d.get(refrgb, _NORGB)
            distance = get_distance(rgb, refrgb)
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
