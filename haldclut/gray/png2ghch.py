# PNG to GHCH converter

# GHCH is a file format of gray hald clut markers


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

    curr_y = 0
    curr_len = 0

    data = bytearray()
    topnibble = True

    for i, rgb in enumerate(img.getdata()):
        assert rgb[0]==rgb[1]==rgb[2]
        y = rgb[0]

        #print([y, curr_y, curr_len, topnibble])

        if y == curr_y:
            if curr_len == 15:
                if topnibble:
                    data.append(curr_len << 4)
                else:
                    data[-1] += curr_len
                    data.append(0)
            else:
                curr_len += 1
        elif curr_len and (y == curr_y + 1 or (curr_y == 255 and y == 0)):
            if topnibble:
                data.append(curr_len << 4)
            else:
                data[-1] += curr_len
            topnibble = not topnibble
            curr_y = y
            curr_len = 1
        else:
            break

    pout = pin.parent / f'{pin.stem}.ghch'
    with pout.open("wb") as fp:
        fp.write(data)
