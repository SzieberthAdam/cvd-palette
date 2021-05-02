import copy
import itertools
import sys

# 3rd party libraries
import colour
import numpy as np
from PIL import Image

def delta_e_from_lab(lab1, lab2):
    delta_E = colour.difference.delta_E_CIE2000(lab1, lab2)
    return delta_E

def ensure_uint8(v):
    return max(0, min(255, int(v)))

def get_lab_arr(rgb_arr):
    xyz_arr = colour.sRGB_to_XYZ(rgb_arr/255)
    lab_arr = colour.XYZ_to_Lab(xyz_arr)
    return lab_arr

def guess_y(n, i):
    x = 255 * i / (n-1)
    y = (
        0.00001351250695406764*x**3
        - 0.00452696095001742487*x**2
        + 1.27471156507023527357*x
        + 1.07967029379810826389
    )  # result of curve fitting
    return ensure_uint8(y)


def get_pal_delta_e(rgb_arr):
    lab_arr = get_lab_arr(rgb_arr)
    combs = list(itertools.combinations(lab_arr, 2))
    lab1_arr, lab2_arr = np.array(list(zip(*combs)))
    delta_e_arr = delta_e_from_lab(lab1_arr, lab2_arr)
    return delta_e_arr.min()

def monopalhexstr(monopal):
    return ", ".join([f'0x{v:0>2X}' for v in monopal])


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f'ERROR! 1<N integer palette size was expected as argument.')
        sys.exit(1)

    try:
        N = int(sys.argv[1])
    except ValueError:
        print(f'ERROR! 1<N integer palette size was expected as argument.')
        sys.exit(1)

    if N <= 1:
        print(f'ERROR! 1<N integer palette size was expected as argument.')
        sys.exit(1)

    start_palette_arr = np.array([guess_y(N, i) for i in range(N)])

    done = False
    max_delta_e = 0
    max_delta_e_pals = set()

    pools = []
    for i, v in enumerate(start_palette_arr):
        if i in (0, N-1):
            pools.append([v])
        else:
            pools.append(sorted({ensure_uint8(v-1), v, ensure_uint8(v+1)}))

    #print(pools)

    nextM = np.array(np.meshgrid(*pools)).T.reshape(-1,N)
    #print(nextM)

    round = 0
    while 0 < nextM.size:
        round += 1
        print(f'Round {round}; {nextM.size} palette candidates')

        M = nextM
        #print(M)
        nextM = np.array([], dtype=int).reshape(-1,N)
        #print(nextM)

        for monopal in M:
            #print(monopal)

            rgb_arr = np.repeat(monopal, 3).reshape((N, 3))
            #print(rgb_arr)

            pal_delta_e = get_pal_delta_e(rgb_arr)

            starstr = (" *" if max_delta_e <= pal_delta_e else "")
            #print(f'[{monopalhexstr(monopal)}] deltaE: {pal_delta_e}{starstr}')

            if max_delta_e < pal_delta_e:
                max_delta_e_pals = set()
            if max_delta_e <= pal_delta_e:
                max_delta_e_pals.add(tuple(monopal))
                max_delta_e = pal_delta_e

        for monopal in max_delta_e_pals:
            for i, y in enumerate(monopal[1:-1], 1):
                #print(i, y, pools[i])
                newval = None
                if 0 < y and y == min(pools[i]):
                    newval = y-1
                    newvalindex = 0
                elif y < 255 and y == max(pools[i]):
                    newval = y+1
                    newvalindex = len(pools[i])
                if newval is not None:
                    nextMpools = copy.deepcopy(pools)
                    nextMpools[i] = [newval]
                    nextM2 = np.array(np.meshgrid(*nextMpools)).T.reshape(-1,N)
                    nextM = np.concatenate((nextM, nextM2))
                    #print(nextM)
                    pools[i].insert(newvalindex, newval)

    print("-"*64)
    print(f'Highest deltaE: {max_delta_e}')

    monoarr = np.array(sorted(max_delta_e_pals))
    for monopal in monoarr:
        print(f'[{monopalhexstr(monopal)}]')


    print()

    with open(f'mono{N:0>2}_deltaE.txt', "w", encoding="utf8") as f:
        f.write(f'{max_delta_e}\n')
    print(f'"mono{N:0>2}_deltaE.txt" saved.')

    arr = np.repeat(monoarr, 3).reshape((1, N, 3)).astype('uint8')
    img = Image.fromarray(arr, 'RGB')
    img.save(f'mono{N:0>2}.png')
    print(f'"mono{N:0>2}.png" saved.')
