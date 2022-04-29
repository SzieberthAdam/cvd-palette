# PNG to GHC converter

# GHC is a compressed file format to describe gray hald cluts


# 3rd party libraries
from PIL import Image

# standard libraries
import pathlib
import sys


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f'ERROR! PNG file was expected as argument.')
        sys.exit(1)

    pin = pathlib.Path(sys.argv[1])
    img = Image.open(pin)

    data = bytearray()

    for i, rgb in enumerate(img.getdata()):
        r, g, b = rgb
        assert r==g==b
        y = r
        if i == 0:
            curr_y, curr_len = y, 1
        elif y == curr_y:
            if curr_len == 254:
                data.extend((curr_y, 255))
                curr_len = 0
            else:
                curr_len += 1
        elif curr_len:
            data.extend((curr_y, curr_len))
            curr_y, curr_len = y, 1
        else:
            curr_y, curr_len = y, 1
    else:
        if curr_len:
            data.extend((curr_y, curr_len))
            curr_len = 0

    pout = pin.parent / f'{pin.stem}.ghc'
    with pout.open("wb") as fp:
        fp.write(data)
