# 3rd party libraries
from PIL import Image

# standard libraries
import pathlib
import sys

def iter_identity_rgb():
    for b in range(256):
        for g in range(256):
            for r in range(256):
                yield r, g, b

def rgbstr(rgb):
    r, g, b = rgb
    return f'#{r:0>2X}{g:0>2X}{b:0>2X}'


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f'ERROR! Image file was expected as argument.')
        sys.exit(1)

    pin = pathlib.Path(sys.argv[1])
    img = Image.open(pin)

    rgbgen = iter_identity_rgb()

    for n, rgb in enumerate(img.getdata(), 1):
        expected_rgb = next(rgbgen)
        assert rgb == expected_rgb, f'pixel {n} differs: {rgbstr(rgb)} != {rgbstr(expected_rgb)} (expected)'
    else:
        print("Identity Hald CLUT confirmed.")
