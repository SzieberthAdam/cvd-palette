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

def grouper(iterable, n):
    args = [iter(iterable)] * n
    yield from (tuple(x for x in t if x is not ...) for t in itertools.zip_longest(*args, fillvalue=...))


def rgbarr_to_labarr(arr):
    arr = colour.sRGB_to_XYZ(arr/255)
    arr = colour.XYZ_to_Lab(arr)
    return arr

def get_boundary_colors(isoluminant_level0_image_path):
    in_img_path = pathlib.Path(isoluminant_level0_image_path)
    in_img = Image.open(str(in_img_path))
    in_rgb_arr = np.array(in_img).reshape((-1, 3))
    in_lab_arr = rgbarr_to_labarr(in_rgb_arr)
    maxmax_dE = None
    maxmax_rgbs = np.array((), dtype="uint8")
    idxs_gen = grouper(itertools.combinations(range(len(in_lab_arr)), 2), 1000000)
    gen = grouper(itertools.combinations(in_lab_arr, 2), 1000000)
    iteration_count = math.comb(in_lab_arr.shape[0], 2)
    print(f'Iterations: {iteration_count}')
    for batchnr, batch in enumerate(gen, 1):
        print(f'B{batchnr}:', end='', flush=True)
        idxs = next(idxs_gen)
        lab1_arr, lab2_arr = np.array(tuple(zip(*batch)))
        print("L", end='', flush=True)
        delta_e_arr = de2000.delta_e_from_lab(lab1_arr, lab2_arr)
        print("D", end='', flush=True)
        max_dE = np.max(delta_e_arr)
        if maxmax_dE is not None and max_dE == maxmax_dE:
            max_idxs = np.where(delta_e_arr == max_dE)
            max_rgbs = np.array(
                [np.array([in_rgb_arr[j] for i in m for j in idxs[i]], dtype="uint8")
                for m in max_idxs], dtype="uint8"
            ).reshape((len(max_idxs), 2, 3))
            maxmax_rgbs = np.concatenate((maxmax_rgbs, max_rgbs))
        elif maxmax_dE is None or maxmax_dE < max_dE:
            maxmax_dE = max_dE
            max_idxs = np.where(delta_e_arr == max_dE)
            max_rgbs = np.array(
                [np.array([in_rgb_arr[j] for i in m for j in idxs[i]], dtype="uint8")
                for m in max_idxs], dtype="uint8"
            ).reshape((len(max_idxs), 2, 3))
            maxmax_rgbs = max_rgbs
        print(".", flush=True)
    return maxmax_dE, maxmax_rgbs


def sort_rgb_by_abl(rgb_arr):
    lab_arr = rgbarr_to_labarr(rgb_arr)
    sort_by_arr = lab_arr[:, [1, 2, 0]]
    del lab_arr
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


def sort_rgb_by_shv(rgb_arr):
    hsv_arr = colour.RGB_to_HSV(rgb_arr / 255)
    sort_by_arr = hsv_arr[:, [1, 0, 2]]
    del hsv_arr
    sort_idx_arr = np.lexsort(sort_by_arr.T[::-1])
    out_rgb_arr = rgb_arr[sort_idx_arr]
    out_sort_by_arr = sort_by_arr[sort_idx_arr]
    return sort_idx_arr, out_rgb_arr, out_sort_by_arr

def sort_rgb_by_s3hsv(rgb_arr):
    hsv_arr = colour.RGB_to_HSV(rgb_arr / 255)
    s3_arr = np.ceil(hsv_arr[:,1] * 4 + 0.0000000000000001).reshape((-1, 1))  # 4 clasters
    sort_by_arr = np.hstack((s3_arr, hsv_arr))
    del hsv_arr, s3_arr
    sort_idx_arr = np.lexsort(sort_by_arr.T[::-1])
    out_rgb_arr = rgb_arr[sort_idx_arr]
    out_sort_by_arr = sort_by_arr[sort_idx_arr]
    return sort_idx_arr, out_rgb_arr, out_sort_by_arr

def sort_rgb_by_s3hsv2(rgb_arr):
    hsv_arr = colour.RGB_to_HSV(rgb_arr / 255)
    s3_arr = np.ceil(hsv_arr[:,1] * 4 + 0.0000000000000001).reshape((-1, 1))  # 4 clasters
    sort_by_arr = np.hstack((s3_arr, hsv_arr))
    del hsv_arr, s3_arr
    ii = np.where(sort_by_arr[:,0] % 2)[0]
    np.put(sort_by_arr[:,1], ii, -sort_by_arr[:,1][ii])
    sort_idx_arr = np.lexsort(sort_by_arr.T[::-1])
    out_rgb_arr = rgb_arr[sort_idx_arr]
    out_sort_by_arr = sort_by_arr[sort_idx_arr]
    return sort_idx_arr, out_rgb_arr, out_sort_by_arr


#sort_rgb = sort_rgb_by_hsv
#sort_rgb = sort_rgb_by_abl
#sort_rgb = sort_rgb_by_shv
#sort_rgb = sort_rgb_by_s3hsv
sort_rgb = sort_rgb_by_s3hsv2


def np_set_difference_argarr(A, B):
    # https://stackoverflow.com/a/40056251/2334951
    # C = [x for x in A if x not in B]
    A, B = in_rgb_arr, source_rgb_arr[0]
    dims = np.maximum(B.max(0),A.max(0), dtype="uint16")+1
    keep = ~np.in1d(np.ravel_multi_index(A.T,dims),np.ravel_multi_index(B.T,dims))
    #C = A[keep]
    return keep


if __name__ == "__main__":

    try:
        graylevel = int(sys.argv[1], 16)
        assert graylevel in range(0,256)
        initcolors = int(sys.argv[2])
    except (IndexError, ValueError, AssertionError):
        print('Usage: python generate.py <graylevel> <initcolors>')
        print('0 <= graylevel <= FF (base 16)')
        sys.exit(1)

    root = pathlib.Path(__file__).parent.resolve()

    in_img_path = root.parent / "isoluminant" / "level0" / f'{graylevel:0>2X}.png'
    isoluminant_level0_image_path = in_img_path
    in_img_path = pathlib.Path(isoluminant_level0_image_path)
    in_img = Image.open(str(in_img_path))
    in_rgb_arr0 = np.array(in_img).reshape((-1, 3))
    # raise Exception   # to test sort functions
    _, in_rgb_arr, in_sortby_arr = sort_rgb(in_rgb_arr0)
    in_lab_arr = rgbarr_to_labarr(in_rgb_arr)
    # raise Exception   # to test sort functions

    bound_img_path = root / f'{graylevel:0>2X}' / f'{graylevel:0>2X}-0000-02.png'
    dEtxtpath = root / f'{graylevel:0>2X}' / f'{graylevel:0>2X}-0000-02.txt'

    if bound_img_path.is_file():
        bound_img = Image.open(str(bound_img_path))
        bound_rgb_arr = np.asarray(bound_img)
        assert bound_rgb_arr.shape[0] == 1  # assert one row, I hope there will never be more
        bound_rgb_arr = bound_rgb_arr.reshape((-1, 3))
        with dEtxtpath.open("r") as f:
            bound_dE = float(f.read())
    else:
        bound_dE, bound_rgb_arr = get_boundary_colors(in_img_path)
        assert bound_rgb_arr.shape[0] == 1  # assert one row, I hope there will never be more
        bound_rgb_arr = bound_rgb_arr.reshape((-1, 3))
        _, bound_rgb_arr, _ = sort_rgb(bound_rgb_arr)
        bound_img = Image.fromarray(bound_rgb_arr.reshape((1, -1, 3)), 'RGB')
        bound_img.save(str(bound_img_path))
        with dEtxtpath.open("w") as f:
            f.write(str(bound_dE))

    source_rgb_arr = bound_rgb_arr

    level = 1
    target_rgb_arr = np.zeros((0, 3))
    colors = source_rgb_arr.shape[0]

    target_img_path = root / f'{graylevel:0>2X}' / f'{graylevel:0>2X}-{initcolors:0>4}-{level:0>2}.png'
    if target_img_path.is_file():
        target_img = Image.open(str(target_img_path))
        target_rgb_arr = np.asarray(target_img).reshape((-1, 3))
        source_rgb_arr = target_rgb_arr
    else:
        while source_rgb_arr.shape[0] < initcolors:
            source_lab_arr = rgbarr_to_labarr(source_rgb_arr)
            #target_rgb_arr = np.copy(source_rgb_arr)

            dE_arr = np.zeros((len(in_lab_arr), source_rgb_arr.shape[0]), dtype="float64")
            for i in range(source_rgb_arr.shape[0]):
                color_lab_arr = np.tile(source_lab_arr[i], len(in_lab_arr)).reshape((len(in_lab_arr), 3))
                dE_arr[:, i] = color_dE_arr = de2000.delta_e_from_lab(in_lab_arr, color_lab_arr)

            values = np.min(dE_arr, axis=1)
            for pick_i in reversed(np.argsort(values)):
                pick_rgb = in_rgb_arr[pick_i]
                print(source_rgb_arr.shape[0] + 1, pick_rgb, values[pick_i])
                #print(pick_rgb, values[pick_i], dE_arr[pick_i])
                target_rgb_arr = np.append(source_rgb_arr, pick_rgb.reshape((1,3)), axis=0)
                _, target_rgb_arr, _ = sort_rgb(target_rgb_arr.reshape((-1, 3)))
                break
            source_rgb_arr = target_rgb_arr

        target_img = Image.fromarray(target_rgb_arr.reshape((1, -1, 3)), 'RGB')
        target_img.save(str(target_img_path))

    while target_rgb_arr.shape[0] < in_rgb_arr.shape[0]:
        level += 1
        target_img_path = root / f'{graylevel:0>2X}' / f'{graylevel:0>2X}-{initcolors:0>4}-{level:0>2}.png'
        if target_img_path.is_file():
            target_img = Image.open(str(target_img_path))
            target_rgb_arr = np.asarray(target_img)
            assert target_rgb_arr.shape[0] == 1  # assert one row, I hope there will never be more
            target_rgb_arr = target_rgb_arr.reshape((-1, 3))
            source_rgb_arr = target_rgb_arr
        else:
            print(f'=== LEVEL {level} ===')
            _, source_rgb_arr, source_sortby_arr = sort_rgb(source_rgb_arr)
            source_lab_arr = rgbarr_to_labarr(source_rgb_arr)
            #print(source_rgb_arr.shape, source_lab_arr.shape, source_sortby_arr.shape)
            target_rgb_arr = np.copy(source_rgb_arr)
            new_rgbs = set()
            for c0 in range(source_sortby_arr.shape[0]):
                c1 = (c0+1) % source_sortby_arr.shape[0]
                c0_rgb = source_rgb_arr[c0]
                c1_rgb = source_rgb_arr[c1]
                c0_lab = source_lab_arr[c0]
                c1_lab = source_lab_arr[c1]
                c0_idx = np.where((in_rgb_arr == c0_rgb).all(axis=1))[0][0]
                c1_idx = np.where((in_rgb_arr == c1_rgb).all(axis=1))[0][0]
                if c0_idx < c1_idx:
                    range0_rgb_arr = in_rgb_arr[c0_idx+1:c1_idx]
                    range0_lab_arr = in_lab_arr[c0_idx+1:c1_idx]
                else:
                    range0_rgb_arr = np.vstack((in_rgb_arr[c0_idx+1:], in_rgb_arr[:c1_idx]))
                    range0_lab_arr = np.vstack((in_lab_arr[c0_idx+1:], in_lab_arr[:c1_idx]))
                n = range0_rgb_arr.shape[0]

                range1_rgb_arrs = [None] * 2
                range1_lab_arrs = [None] * 2
                if n == 0:
                    continue
                elif n == 1:
                    range1_rgb_arrs[0] = range0_rgb_arr[:1]
                    range1_rgb_arrs[1] = None # np.array([], shape=(0, 3), dtype="uint8")
                    range1_lab_arrs[0] = range0_lab_arr[:1]
                    range1_lab_arrs[1] = None # np.array([], shape=(0, 3), dtype="float64")
                elif n in {2, 3}:
                    range1_rgb_arrs[0] = range0_rgb_arr[:1]
                    range1_rgb_arrs[1] = range0_rgb_arr[-1:]
                    range1_lab_arrs[0] = range0_lab_arr[:1]
                    range1_lab_arrs[1] = range0_lab_arr[-1:]
                elif n in {4, 5}:
                    range1_rgb_arrs[0] = range0_rgb_arr[:2]
                    range1_rgb_arrs[1] = range0_rgb_arr[-2:]
                    range1_lab_arrs[0] = range0_lab_arr[:2]
                    range1_lab_arrs[1] = range0_lab_arr[-2:]
                else:
                    r = n//6
                    range1_rgb_arrs[0] = range0_rgb_arr[r:2*r]
                    range1_rgb_arrs[1] = range0_rgb_arr[-2*r:-r]
                    range1_lab_arrs[0] = range0_lab_arr[r:2*r]
                    range1_lab_arrs[1] = range0_lab_arr[-2*r:-r]

                cc_rgbs = (c0_rgb, c1_rgb)
                cc_labs = (c0_lab, c1_lab)
                for cci in range(2):
                    if range1_rgb_arrs[cci] is None:
                        continue
                    elif range1_rgb_arrs[cci].shape[0] == 1:
                        color_rgb_pick = range1_rgb_arrs[cci][0]
                    else:
                        color_lab_arr = np.tile(cc_labs[cci], range1_lab_arrs[cci].shape[0]).reshape((-1, 3))
                        color_dE_arr = de2000.delta_e_from_lab(range1_lab_arrs[cci], color_lab_arr)
                        color_dE_arr_ii = np.argsort(color_dE_arr)
                        color_dE_arr_pick_ii = color_dE_arr_ii[color_dE_arr_ii.shape[0]//2]
                        color_rgb_pick = range1_rgb_arrs[cci][color_dE_arr_pick_ii]
                    target_rgb_arr = np.append(target_rgb_arr, color_rgb_pick).reshape((-1, 3))


                    #dE_arrs = [np.zeros(range1_lab_arrs[i].shape[0], dtype="float64") for i in range(2)]

            _, target_rgb_arr, _ = sort_rgb(target_rgb_arr)

            target_img = Image.fromarray(target_rgb_arr.reshape((1, -1, 3)), 'RGB')
            target_img.save(str(target_img_path))

            source_rgb_arr = target_rgb_arr

            #input("PRESS ENTER TO CONTINUE")
