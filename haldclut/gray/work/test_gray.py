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

QUANTUM_RANGE = 255
QUANTUM_SCALE = 1 / QUANTUM_RANGE


def method_rgb(rgb, method):
    method = (method or 'none').lower()
    if method == 'none':
        return rgb
    elif method == 'luma':
        return tuple(encodepixelgamma(x) for x in rgb)
    elif method == 'luminance':
        return tuple(decodepixelgamma(x) for x in rgb)


def method_y(y, method):
    return y
    #return decodepixelgamma(y)
    #return encodepixelgamma(y)
    method = (method or 'none').lower()
    if method == 'none':
        return y
    elif method == 'luma':
        return decodepixelgamma(y)
    elif method == 'luminance':
        return encodepixelgamma(y)


def clamptoquantum(quantum):
    if quantum <= 0:
        return 0
    elif QUANTUM_RANGE <= quantum:
        return QUANTUM_RANGE
    return int(quantum+0.5)

def decodegamma(x):
    return x**2.4

def decodepixelgamma(pixel):
    if pixel <= 0.0404482362771076 * QUANTUM_RANGE:
        return pixel / 12.92
    else:
        return QUANTUM_RANGE * decodegamma((QUANTUM_SCALE * pixel + 0.055)/1.055)

def encodegamma(x):
    return x**(1.0/2.4)

def encodepixelgamma(pixel):
    if pixel <= 0.0031306684425005883 * QUANTUM_RANGE:
        return pixel * 12.92
    else:
        return QUANTUM_RANGE * (1.055 * encodegamma(QUANTUM_SCALE * pixel) - 0.055)

def y_itu_bt_601(rgb, method=None):
    r, g, b = method_rgb(rgb, method)
    cr = 0.299
    cg = 0.587
    cb = 0.114
    y = cr*r + cg*g + cb*b
    return clamptoquantum(method_y(y, method))

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

def y_itu_bt_709(rgb, method=None):
    r, g, b = method_rgb(rgb, method)
    cr = 0.212656 # 0.212655
    cg = 0.715158
    cb = 0.072186 # 0.072187
    y = cr*r + cg*g + cb*b
    return clamptoquantum(method_y(y, method))



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
    diters[f'ITU BT.601 (none)'] = y_itu_bt_601
    diters[f'ITU BT.709 (none)'] = y_itu_bt_709
    diters[f'ITU BT.601 (luma)'] = partial(y_itu_bt_601, method="luma")
    diters[f'ITU BT.709 (luma)'] = partial(y_itu_bt_709, method="luma")
    diters[f'ITU BT.601 (luminance)'] = partial(y_itu_bt_601, method="luminance")
    diters[f'ITU BT.709 (luminance)'] = partial(y_itu_bt_709, method="luminance")

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
