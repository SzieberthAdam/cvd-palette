import itertools
import math
import pathlib
import sys

_thisdir = pathlib.Path(__file__).parent.resolve()
_importdir = _thisdir.parent.resolve()
sys.path.insert(0, str(_importdir))

# 3rd party libraries (module file available)
import de2000

sys.path.remove(str(_importdir))

# 3rd party libraries
import colour
from PIL import Image
import numpy as np


def rgbarr_to_labarr(arr):
    arr = colour.sRGB_to_XYZ(arr/255)
    arr = colour.XYZ_to_Lab(arr)
    return arr
    

def sort_rgb_by_rgb(rgb_arr):
    sort_by_arr = rgb_arr
    sort_idx_arr = np.lexsort(sort_by_arr.T[::-1])
    out_rgb_arr = rgb_arr[sort_idx_arr]
    out_sort_by_arr = sort_by_arr[sort_idx_arr]
    return sort_idx_arr, out_rgb_arr, out_sort_by_arr


def sort_rgb_by_hsv(rgb_arr):
    sort_by_arr = colour.RGB_to_HSV(rgb_arr / 255)
    # https://stackoverflow.com/questions/38277143
    # I would like to sort it lexicographically, i.e. by first column, then by second column, and so on until
    # the last column. This is what numpy.lexsort is for, but the interface is awkward. Pass it a 2D array, and
    # it will argsort the columns, sorting by the last row first, then the second-to-last row, continuing up to
    # the first row. If you want to sort by rows, with the first column as the primary key, you need to rotate
    # the array before lexsorting it.
    # sort_idx_arr = np.lexsort(np.rot90(sort_by_arr))
    # One could add that there's a more time-efficient way of getting the same result as with rot90, by using
    # x[numpy.lexsort(x.T[::-1])]. According to timeit, this is about 25% faster than
    # x[numpy.lexsort(numpy.rot90(x))]
    sort_idx_arr = np.lexsort(sort_by_arr.T[::-1])
    out_rgb_arr = rgb_arr[sort_idx_arr]
    out_sort_by_arr = sort_by_arr[sort_idx_arr]
    return sort_idx_arr, out_rgb_arr, out_sort_by_arr


#sort_rgb = sort_rgb_by_rgb
sort_rgb = sort_rgb_by_hsv


if __name__ == "__main__":

    out_img_path = _thisdir / "ilum256.png"
    #out_txt_path = _thisdir / "ilum256.txt"
    
    if out_img_path.is_file():
        out_img = Image.open(str(out_img_path))
        out_img_arr = np.array(out_img)
    else:
        out_img_arr = np.zeros((256, 256, 3), dtype="uint8")
    
    try:
        graylevels = []
        for x in sys.argv[1:]:
            if "-" in x:
                s, e = x.split("-")
                s = s or "0"
                e = e or "255"
            elif "–" in x:
                s, e = x.split("–")
                s = s or "0"
                e = e or "255"
            else:
                s = x
                e = None
            s = int(s)
            assert s in range(0,256)
            if e is None:
                graylevels.append(s)
            else:
                e = int(e)
                assert e in range(0,256)
                if s <= e:
                    graylevels.extend(range(s, e+1))
                else:
                    graylevels.extend(range(s, e-1, -1))
    except (IndexError, ValueError, AssertionError):
        print('Usage: python generate256x256.py [graylevel]...')
        print('0 <= graylevel <= FF (base 16)')
        print('graylevel range format: 00-FF')
        sys.exit(1)
    if not graylevels:
        graylevels = list(range(256))

    root = pathlib.Path(__file__).parent.resolve()

    for graylevel in graylevels:

        print(f'=== {graylevel:0>3} ===')

        in_img_path = root.parent / "isoluminant" / "szieberth" / f'{graylevel:0>2x}.png'
        isoluminant_image_path = in_img_path
        in_img_path = pathlib.Path(isoluminant_image_path)
        in_img = Image.open(str(in_img_path))
        in_rgb_arr0 = np.array(in_img).reshape((-1, 3))
        # raise Exception   # to test sort functions
        #_, in_rgb_arr, in_sortby_arr = sort_rgb(in_rgb_arr0)
        in_rgb_arr = in_rgb_arr0
        in_lab_arr = rgbarr_to_labarr(in_rgb_arr)
        # raise Exception   # to test sort functions

        graylevel_dir = root / f'{graylevel:0>2x}'
        if not graylevel_dir.is_dir():
            graylevel_dir.mkdir(parents=True, exist_ok=True)

        bound_img_path = graylevel_dir / f'{graylevel:0>2x}-0002-01.png'

        bound_img = Image.open(str(bound_img_path))
        bound_rgb_arr = np.asarray(bound_img)
        assert bound_rgb_arr.shape[0] == 1  # assert one row, I hope there will never be more
        bound_rgb_arr = bound_rgb_arr.reshape((-1, 3))
        _, bound_rgb_arr, _ = sort_rgb(bound_rgb_arr)

        out_img_arr[graylevel][:2] = bound_rgb_arr
    
        source_rgb_arr = bound_rgb_arr
    
        for c in range(2, 256):
            source_lab_arr = rgbarr_to_labarr(source_rgb_arr)
            
            dE_arr = np.zeros((len(in_lab_arr), source_rgb_arr.shape[0]), dtype="float64")
            for i in range(source_rgb_arr.shape[0]):
                color_lab_arr = np.tile(source_lab_arr[i], len(in_lab_arr)).reshape((len(in_lab_arr), 3))
                dE_arr[:, i] = color_dE_arr = de2000.delta_e_from_lab(in_lab_arr, color_lab_arr)
        
            values = np.min(dE_arr, axis=1)
            for pick_i in reversed(np.argsort(values)):
                pick_rgb = in_rgb_arr[pick_i]
                print(source_rgb_arr.shape[0] + 1, pick_rgb, values[pick_i])
                #print(pick_rgb, values[pick_i], dE_arr[pick_i])
                out_img_arr[graylevel][c] = pick_rgb.reshape((1,3))
                break
           
            source_rgb_arr = out_img_arr[graylevel][:c+1]
    
        out_img = Image.fromarray(out_img_arr, 'RGB')
        out_img.save(str(out_img_path))
