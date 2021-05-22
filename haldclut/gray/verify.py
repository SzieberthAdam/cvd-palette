# 3rd party libraries
from PIL import Image
import numpy as np

# standard libraries
import pathlib
import sys

_thisdir = pathlib.Path(__file__).parent.resolve()

grayclut_paths = sorted(_thisdir.glob("gray.*.png"))

def rgbstr(rgb):
    r, g, b = rgb
    return f'#{r:0>2X}{g:0>2X}{b:0>2X}'

if __name__ == "__main__":

    if len(sys.argv) == 2:
        grayclut_paths = [pathlib.Path(sys.argv[1])]
    else:
        grayclut_paths = sorted(_thisdir.glob("gray.*.png"))

    logparts = []

    refcheck = {1, 2}

    for p in grayclut_paths:

        prevRGB = (256, 256, 256)
        check = set(refcheck)

        name = p.stem[5:]
        img = Image.open(str(p))
        arr = np.array(img).reshape((-1,3))

        for n, RGB in enumerate(arr, 1):
            if not check:
                break
            RGB = tuple(RGB)
            RGBs, prevRGBs = rgbstr(RGB), rgbstr(prevRGB)
            if 1 in check:
                if not RGB[0] == RGB[1] == RGB[2]:
                    s = f'FAIL1: {name} has non-gray {rgbstr(RGB)} at pixel {n}.'
                    logparts.append(s)
                    print(s)
                    check.remove(1)
            if 2 in check:
                if (n-1) % 256:
                    if not all(prevRGB[j] <= RGB[j] for j in range(3)):
                        s = f'FAIL2D: {name} has decreasing gray sequence {prevRGBs}, {RGBs} at pixel {n}.'
                        logparts.append(s)
                        print(s)
                        check.remove(2)
                else:
                    if not all(RGB[j] < prevRGB[j] for j in range(3)):
                        s = f'FAIL2N: {name} has non-decreasing gray sequence {prevRGBs}, {RGBs} at pixel {n}.'
                        logparts.append(s)
                        print(s)
                        check.remove(2)
            prevRGB = RGB
        else:
            if check == refcheck:
                s = f'OK: {name}'
                logparts.append(s)
                print(s)

    p = _thisdir / "verify.log"
    with p.open("w") as f:
        f.write("\n".join(logparts) + "\n")

    print(f'"{p}" saved.')
