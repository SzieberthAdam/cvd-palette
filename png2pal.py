
import pal
import pathlib

from PIL import Image

import sys


def main(filename=None, *, verbose=False):
    if filename is None:
        if verbose:
            print(f'ERROR! PNG file was expected as argument.')
        return 1

    pin = pathlib.Path(filename)
    img = Image.open(pin)
    data = tuple(img.getdata())

    pout = pin.resolve().parent / f'{pin.stem}.pal'
    with pout.open("w", encoding="ascii") as fp:
        pal.dump(data, fp)

    if verbose:
        print(f'"{pin.stem}.pal" saved.')

    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:], verbose=True))


