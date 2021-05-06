
import pal
import pathlib
import sys

def rgbstr(rgb):
    r, g, b = rgb
    return f'{r:0>2X}{g:0>2X}{b:0>2X}'

def hexstr(palarr):
    return "\n".join(rgbstr(rgb) for rgb in palarr)


def dump(pal_arr, fp):
    return fp.write(hexstr(pal_arr) + "\n")


def main(filename=None, *, verbose=False):
    if filename is None:
        if verbose:
            print(f'ERROR! PAL file was expected as argument.')
        return 1

    pin = pathlib.Path(filename)
    with pin.open() as fp:
        data = pal.load(fp)

    pout = pin.resolve().parent / f'{pin.stem}.hex'

    with pout.open("w", encoding="ascii") as fp:
        dump(data, fp)

    if verbose:
        print(f'"{pin.stem}.hex" saved.')

    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:], verbose=True))
