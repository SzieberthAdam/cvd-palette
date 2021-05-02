
import pal
import pathlib

from PIL import Image

import sys


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f'ERROR! PAL file was expected as argument.')
        sys.exit(1)

    pin = pathlib.Path(sys.argv[1])
    with pin.open() as fp:
        data = pal.load(fp)

    img = Image.new('RGB', [len(data), 1])
    img.putdata([tuple(rgb) for rgb in data])
    # Note: ridiculously RGB values as lists are not accepted only as tuples

    pout = pin.resolve().parent / f'{pin.stem}.png'
    img.save(pout)

    print(f'"{pin.stem}.png" saved.')
