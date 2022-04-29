# 3rd party libraries
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

# standard libraries
import collections
from functools import partial
import pathlib
import random
import sys


if __name__ == "__main__":

    _thisdir = pathlib.Path(__file__).parent.resolve()
    _importdir = _thisdir.parent.parent.resolve()
    sys.path.insert(0, str(_importdir))

    # 3rd party libraries (module file available)
    import clut

    sys.path.remove(str(_importdir))

    img0path = pathlib.Path(sys.argv[1])
    img1path = pathlib.Path(sys.argv[2])

    img0 = Image.open(str(img0path))
    img1 = Image.open(str(img1path))

    log = []

    c = 0
    for x in range(img0.width):
        for y in range(img0.height):
            px0 = img0.getpixel((x,y))
            px1 = img1.getpixel((x,y))
            #print(f'{x},{y} {px0} {px1}')
            if px0 != px1:
                c += 1
                log.append(f'diff #{c} at coord {x},{y}: {px0} != {px1}')
                print(log[-1])
    with open("compare2gray.log", "w", encoding="utf8") as f:
        f.write("\n".join(log))
