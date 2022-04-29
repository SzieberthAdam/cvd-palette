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

def main(filename=None, *, verbose=False):
    if filename is None:
        if verbose:
            print(f'ERROR! PNG file was expected as argument.')
        return 1

    pv = 0
    p = pathlib.Path(str(filename)).resolve()
    img = Image.open(str(p))
    arr = np.array(img)

    for y, row_arr in enumerate(arr):
        for x, RGB_arr in enumerate(row_arr):
            xmod256 = x % 256
            v = RGB_arr[0]
            if xmod256:
                if v < pv:
                    ho = 1  # hole
                    while (x-ho-1) % 256 < xmod256 and v < row_arr[x-ho-1][0]:
                        ho += 1

                    hi = 1  # hill
                    while (x+hi) % 256 and row_arr[x+hi][0] < pv:
                        hi += 1

                    if ho <= hi:
                        vh0 = row_arr[x-ho-1][0]
                        for i in range(ho):
                            vh1 = vh0 + round((v-vh0)/(ho+1)*(i+1))
                            row_arr[x-ho+i] = vh1, vh1, vh1
                            print(f'hole {i+1}/{ho}, X={x-ho+i} Y={y} <- {vh1:0>2X}')
                    else:
                        vh0 = row_arr[x+hi][0]
                        for i in range(hi):
                            vh1 = pv + round((vh0-pv)/(hi+1)*(i+1))
                            row_arr[x+i] = vh1, vh1, vh1
                            print(f'hill {i+1}/{hi}, X={x+i} Y={y} <- {vh1:0>2X}')
                pv = v
                #print(f'a {pv}')
            else:
                pv = 0
                #print(f'b {pv}')
    img1 = Image.fromarray(arr, 'RGB')
    img1.save(p.parent / f'{p.stem}.fixedholes{p.suffix}')
    return 0

if __name__ == "__main__":

    sys.exit(main(*sys.argv[1:], verbose=True))
