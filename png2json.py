
import json
import pathlib

from PIL import Image

import sys


def rgbstr(rgb):
    r, g, b = rgb
    return f'#{r:0>2X}{g:0>2X}{b:0>2X}'


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f'ERROR! PNG file was expected as argument.')
        sys.exit(1)

    pin = pathlib.Path(sys.argv[1])
    img = Image.open(pin)
    data = list(rgbstr(rgb) for rgb in img.getdata())

    pout = pin.resolve().parent / f'{pin.stem}.json'
    with pout.open("w", encoding="ascii") as fp:
        json.dump(data, fp)

    print(f'"{pin.stem}.json" saved.')
