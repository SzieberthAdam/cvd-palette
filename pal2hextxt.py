
import pal
import pathlib
import sys

def rgbstr(rgb):
    r, g, b = rgb
    return f'#{r:0>2X}{g:0>2X}{b:0>2X}'

def palstr(palarr):
    return ", ".join(rgbstr(rgb) for rgb in palarr)


def main(filename=None, *, verbose=False):
    if filename is None:
        if verbose:
            print(f'ERROR! PAL file was expected as argument.')
        return 1

    pin = pathlib.Path(filename)
    with pin.open() as fp:
        data = pal.load(fp)

    pout = pin.resolve().parent / f'{pin.stem}.hex.txt'

    with pout.open("w", encoding="ascii") as fp:
        fp.write(palstr(data)+ "\n")

    if verbose:
        print(f'"{pin.stem}.hex.txt" saved.')

    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:], verbose=True))
