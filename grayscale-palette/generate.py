import copy
import itertools
import pathlib
import sys

from decimal import Decimal, getcontext
getcontext().prec = 30

# 3rd party libraries
import colour
import numpy as np
from PIL import Image

_thisdir = pathlib.Path(__file__).parent.resolve()
_parentdir = _thisdir.parent.resolve()
sys.path.insert(0, str(_parentdir))

# 3rd party libraries (module file available)
import de2000

sys.path.remove(str(_parentdir))

def ensure_uint8(v):
    return max(0, min(255, int(v)))

def guess_y_for_x(x, rounded=True):
    y = (
        - Decimal("0.00000000039293428346")*x**5
        + Decimal("0.00000027548158813178")*x**4
        - Decimal("0.00005902279731056096")*x**3
        + Decimal("0.00418862809152376355")*x**2
        + Decimal("0.82592783724614506785")*x
        + Decimal("9.01889824926381081760")
    )  # result of curve fitting
    if rounded:
        y = ensure_uint8(int(round(y, 0)))
    return y

def guess_y(n, i, rounded=True):
    if i == 0:
        return 0
    elif i == n - 1:
        return 255
    x = Decimal(255 * i) / (n-1)
    return guess_y_for_x(x, rounded=rounded)

def grayscalepalhexstr(grayscalepal):
    return ", ".join([f'0x{v:0>2X}' for v in grayscalepal])


def getn(pools):
    value = 1
    for L in pools:
        value *= len(L)
    return value


def main(*argv):
    if len(argv) < 2:
        print(f'ERROR! 1<N integer palette size was expected as argument.')
        return 1

    try:
        N = int(argv[1])
    except ValueError:
        print(f'ERROR! 1<N integer palette size was expected as argument.')
        return 1

    if N <= 1:
        print(f'ERROR! 1<N integer palette size was expected as argument.')
        return 1

    start_palette_arr = np.array([guess_y(N, i) for i in range(N)])

    print(f'Initial guess:  [{grayscalepalhexstr(start_palette_arr)}]')

    done = False
    max_delta_e = (0,)
    max_delta_e_pals = None

    pools = []
    for i, v in enumerate(start_palette_arr):
        if i in (0, N-1):
            pools.append([v])
        else:
            pools.append(sorted({ensure_uint8(v-1), v, ensure_uint8(v+1)}))

    #print(pools)

    new_palette_candidates = itertools.product(*pools)
    #print(new_palette_candidates)

    n = getn(pools)
    round = 0
    while 0 < n:
        round += 1
        k = 0
        print(f'Round {round}; {n} palette candidates')

        current_palette_candidates = new_palette_candidates
        #print(current_palette_candidates)
        new_palette_candidates = []
        n = 0
        #print(new_palette_candidates)

        for grayscalepal in current_palette_candidates:
            #print(grayscalepal)

            rgb_arr = np.repeat(grayscalepal, 3).reshape((N, 3))
            #print(rgb_arr)

            pal_delta_e = de2000.get_pal_delta_e(rgb_arr)

            #starstr = (" *" if max_delta_e <= pal_delta_e else "")
            #print(f'[{grayscalepalhexstr(grayscalepal)}] deltaE: {pal_delta_e}{starstr}')

            if max_delta_e < pal_delta_e:
                max_delta_e_pals = set()
            if max_delta_e <= pal_delta_e:
                max_delta_e_pals.add(tuple(grayscalepal))
                max_delta_e = pal_delta_e

            k += 1
            if k % 100000 == 0:
                print(f'at {k} iteration; max delta E: {max_delta_e[0]}')

        for grayscalepal in max_delta_e_pals:
            for i, y in enumerate(grayscalepal[1:-1], 1):
                #print(i, y, pools[i])
                newval = None
                if 0 < y and y == min(pools[i]):
                    newval = y-1
                    newvalindex = 0
                elif y < 255 and y == max(pools[i]):
                    newval = y+1
                    newvalindex = len(pools[i])
                if newval is not None:
                    new_pools = copy.deepcopy(pools)
                    new_pools[i] = [newval]
                    new_palette_candidates2 = itertools.product(*new_pools)
                    new_palette_candidates = itertools.chain(new_palette_candidates, new_palette_candidates2)
                    n += getn(new_pools)
                    #print(new_palette_candidates)
                    pools[i].insert(newvalindex, newval)

    print("-"*64)
    print(f'Highest deltaE: {max_delta_e[0]}')

    grayscalearr = np.array(sorted(max_delta_e_pals))
    for grayscalepal in grayscalearr:
        print(f'[{grayscalepalhexstr(grayscalepal)}]')
        print()
        rgb_arr = np.repeat(grayscalepal, 3).reshape((N, 3))
        with open(f'grayscale{N:0>2}_deltaE.txt', "w", encoding="utf8") as f:
            f.write(de2000.get_pal_delta_e_pairs_report(rgb_arr))
        print(f'"grayscale{N:0>2}_deltaE.txt" saved.')

    arr = np.repeat(grayscalearr, 3).reshape((len(grayscalearr), N, 3)).astype('uint8')
    img = Image.fromarray(arr, 'RGB')
    img.save(f'grayscale{N:0>2}.png')
    print(f'"grayscale{N:0>2}.png" saved.')

    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv))
