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


def inv_gam_sRGB(v):
    v /= 255
    if v <= 0.04045:
        return v / 12.92
    else:
        return ((v+0.055)/1.055)**2.4

def gam_sRGB(y):
    if y <= 0.0031308:
        y *= 12.92
    else:
        y = 1.055*y**(1.0/2.4)-0.055
    return round(y*255)

def y_itu_bt_601(rgb, linearize=True, endround=False):
    if linearize:
        r, g, b = [inv_gam_sRGB(x) for x in rgb]
    else:
        r, g, b = rgb
    cr = 0.299
    cg = 0.587
    cb = 0.114
    y = cr*r + cg*g + cb*b
    if linearize:
        y = gam_sRGB(y)
    if endround:
        y = round(y)
    return y

#def y_itu_bt_601(rgb):
#    r, g, b = D(rgb[0]), D(rgb[1]), D(rgb[2])
#    lr, lg, lb = r/255, g/255, b/255
#    cr = D("0.299")
#    cg = D("0.587")
#    cb = D("0.114")
#    ly = cr*lr + cg*lg + cb*lb
#    y = 255*ly
#    y = round(y)
#    return y

def y_itu_bt_709(rgb, linearize=True, endround=False):
    if linearize:
        r, g, b = [inv_gam_sRGB(x) for x in rgb]
    else:
        r, g, b = rgb
    cr = 0.212655
    cg = 0.715158
    cb = 0.072187
    y = cr*r + cg*g + cb*b
    if linearize:
        y = gam_sRGB(y)
    if endround:
        y = round(y)
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
    diters[f'ITU BT.601'] = y_itu_bt_601
    diters[f'ITU BT.709'] = y_itu_bt_709
    diters[f'ITU BT.601 (wrong, non-linearized)'] = partial(y_itu_bt_601, linearize=False, endround=True)
    diters[f'ITU BT.709 (wrong, non-linearized)'] = partial(y_itu_bt_709, linearize=False, endround=True)


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
