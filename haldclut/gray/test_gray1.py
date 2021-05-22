# 3rd party libraries
from PIL import Image

# standard libraries
import collections
from decimal import Decimal as D, getcontext
from functools import partial
import pathlib
import sys

# same direcotry
import test_identity


DEFAULT_DPREC = 20


def y_itu_bt_601(rgb, cprec=3, endround=True):
    getcontext().prec = cprec
    r, g, b = D(rgb[0]), D(rgb[1]), D(rgb[2])
    lr, lg, lb = r/255, g/255, b/255
    q = D("1." + cprec * "0")
    cr = D("0.299").quantize(q)
    cg = D("0.587").quantize(q)
    cb = D("0.114").quantize(q)
    ly = cr*lr + cg*lg + cb*lb
    y = 255*ly
    if endround:
        y = round(y)
    getcontext().prec = DEFAULT_DPREC
    return y

def y_itu_bt_709(rgb, cprec=4, endround=True):
    getcontext().prec = cprec
    r, g, b = D(rgb[0]), D(rgb[1]), D(rgb[2])
    lr, lg, lb = r/255, g/255, b/255
    q = D("1." + cprec * "0")
    cr = D("0.2126").quantize(q)
    cg = D("0.7152").quantize(q)
    cb = D("0.0722").quantize(q)
    ly = cr*lr + cg*lg + cb*lb
    y = 255*ly
    if endround:
        y = round(y)
    getcontext().prec = DEFAULT_DPREC
    return y


if __name__ == "__main__":

    if len(sys.argv) == 3:
        tolerance = int(sys.argv[1])
        pin = pathlib.Path(sys.argv[2])
    elif len(sys.argv) < 2:
        print(f'ERROR! PNG file was expected as argument.')
        sys.exit(1)
    else:
        tolerance = 0
        pin = pathlib.Path(sys.argv[1])
    img = Image.open(pin)

    identity_rgbs = tuple(test_identity.iter_identity_rgb())

    diters = {}

    for cprec in range(1,4):
        fn = partial(y_itu_bt_601, cprec=cprec)
        diters[f'ITU BT.601 (prec={cprec})'] = fn

    for cprec in range(1,5):
        fn = partial(y_itu_bt_709, cprec=cprec)
        diters[f'ITU BT.709 (prec={cprec})'] = fn

    remained_methods = set(diters.keys())
    differences = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))

    for n, rgb in enumerate(img.getdata(), 1):
        assert rgb[0] == rgb[1] == rgb[2], "Not a grayscale image!"
        y = rgb[0]
        identity_rgb = identity_rgbs[n-1]
        last_remained_methods = set(remained_methods)
        for k in sorted(remained_methods):
            fn = diters[k]
            exp_y = fn(identity_rgb)
            exp_rgb = exp_y, exp_y, exp_y
            diff = abs(y-exp_y)
            if tolerance < diff:
                remained_methods.remove(k)
            if diff:
                differences[k][diff] += 1
                #print(f'{k} difference={diff} at pixel {n} ({test_identity.rgbstr(identity_rgb)}).')
                #print(f'{test_identity.rgbstr(rgb)} != {test_identity.rgbstr(exp_rgb)} (expected)')
        if not remained_methods:
            print(f'No method was found at pixel {n} ({test_identity.rgbstr(identity_rgb)}).')
            print(f'{test_identity.rgbstr(rgb)} != {test_identity.rgbstr(exp_rgb)} (expected)')
            for k in sorted(last_remained_methods):
                print(f'Last method: {k}')
            sys.exit(100)
            break
    else:
        for k in remained_methods:
            print(f'Method {k} is confirmed.')
            if differences[k]:
                for di in sorted(differences[k], reverse=True):
                    print(f'Difference {di} count: {differences[k][di]}')
