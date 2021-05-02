
import pal
import pathlib

from PIL import Image

import sys


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f'ERROR! PNG file was expected as argument.')
        sys.exit(1)

    pin = pathlib.Path(sys.argv[1])
    img = Image.open(pin)
    data = tuple(img.getdata())

    pout = pin.resolve().parent / f'{pin.stem}.pal'
    with pout.open("w", encoding="ascii") as fp:
        pal.dump(data, fp)

    print(f'"{pin.stem}.pal" saved.')
