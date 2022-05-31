import itertools
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

dEbound = 10

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
    for batch in gen:
        idxs = next(idxs_gen)
        lab1_arr, lab2_arr = np.array(tuple(zip(*batch)))
        delta_e_arr = de2000.delta_e_from_lab(lab1_arr, lab2_arr)
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
    return maxmax_dE, maxmax_rgbs


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
    isoluminant_level0_image_path = in_img_path # devtime
    in_img_path = pathlib.Path(isoluminant_level0_image_path) # devtime
    in_img = Image.open(str(in_img_path)) # devtime
    in_rgb_arr = np.array(in_img).reshape((-1, 3)) # devtime
    in_lab_arr = rgbarr_to_labarr(in_rgb_arr) # devtime



    bound_img_path = root / f'{graylevel:0>2X}' / f'{graylevel:0>2X}-0000-02.png'
    dEtxtpath = root / f'{graylevel:0>2X}' / f'{graylevel:0>2X}-0000-02.txt'

    if bound_img_path.is_file():
        bound_img = Image.open(str(bound_img_path))
        bound_rgb_arr = np.asarray(bound_img)
        with dEtxtpath.open("r") as f:
            bound_dE = float(f.read())
    else:
        bound_dE, bound_rgb_arr = get_boundary_colors(in_img_path)
        bound_img = Image.fromarray(bound_rgb_arr, 'RGB')
        bound_img.save(str(bound_img_path))
        with dEtxtpath.open("w") as f:
            f.write(str(bound_dE))

    level = 1
    source_rgb_arr = bound_rgb_arr

    test_tup = tuple(map(tuple, source_rgb_arr[0]))
    del_i = tuple(i for i, a in enumerate(in_rgb_arr) if tuple(a) in test_tup)
    print(del_i)
    in_rgb_arr = np.delete(in_rgb_arr, del_i, axis=0).reshape((-1, 3))
    in_lab_arr = np.delete(in_lab_arr, del_i, axis=0).reshape((-1, 3))

    target_rgb_arr = np.zeros((1, 0, 3))

    colors = source_rgb_arr.shape[1]

    target_img_path = root / f'{graylevel:0>2X}' / f'{graylevel:0>2X}-{initcolors:0>4}-{level:0>2}.png'
    if target_img_path.is_file():
        target_img = Image.open(str(target_img_path))
        target_rgb_arr = np.asarray(target_img)
        source_rgb_arr = target_rgb_arr
        test_tup = tuple(map(tuple, source_rgb_arr[0]))
        del_i = tuple(i for i, a in enumerate(in_rgb_arr) if tuple(a) in test_tup)
        in_rgb_arr = np.delete(in_rgb_arr, del_i, axis=0).reshape((-1, 3))
        in_lab_arr = np.delete(in_lab_arr, del_i, axis=0).reshape((-1, 3))
    else:
        while source_rgb_arr.shape[1] < initcolors:
            source_lab_arr = rgbarr_to_labarr(source_rgb_arr)
            #target_rgb_arr = np.copy(source_rgb_arr)

            dE_arr = np.zeros((len(in_lab_arr), source_rgb_arr.shape[1]), dtype="float64")
            for i in range(source_rgb_arr.shape[1]):
                color_lab_arr = np.tile(source_lab_arr[0][i], len(in_lab_arr)).reshape((len(in_lab_arr), 3))
                dE_arr[:, i] = color_dE_arr = de2000.delta_e_from_lab(in_lab_arr, color_lab_arr)

            filtr = np.all(dEbound <= dE_arr, axis=1)
            if not np.any(filtr):
                break
            values = np.min(dE_arr, axis=1)
            for pick_i in reversed(np.argsort(values)):
                if not filtr[pick_i]:
                    continue
                pick_rgb = in_rgb_arr[pick_i]
                print(pick_rgb, values[pick_i])
                #print(pick_rgb, values[pick_i], dE_arr[pick_i])
                target_rgb_arr = np.append(source_rgb_arr, pick_rgb.reshape((1, 1, 3)), axis=1)
                target_rgb_arr_shape = target_rgb_arr.shape
                target_hsv_arr = colour.RGB_to_HSV(target_rgb_arr / 255)[0]
                argsort_by_hues = np.argsort(target_hsv_arr, axis=0)[:,0]
                target_rgb_arr = (target_rgb_arr[0][argsort_by_hues]).reshape(target_rgb_arr_shape)
                break
            source_rgb_arr = target_rgb_arr
            test_tup = tuple(map(tuple, source_rgb_arr[0]))
            del_i = tuple(i for i, a in enumerate(in_rgb_arr) if tuple(a) in test_tup)
            in_rgb_arr = np.delete(in_rgb_arr, del_i, axis=0).reshape((-1, 3))
            in_lab_arr = np.delete(in_lab_arr, del_i, axis=0).reshape((-1, 3))

        target_img = Image.fromarray(target_rgb_arr, 'RGB')
        target_img.save(str(target_img_path))

    while 0 < in_rgb_arr.shape[0]:
        level += 1
        in_hue_arr = colour.RGB_to_HSV(in_rgb_arr.reshape((1, -1, 3)) / 255)[0][:,0]
        target_img_path = root / f'{graylevel:0>2X}' / f'{graylevel:0>2X}-{initcolors:0>4}-{level:0>2}.png'
        if target_img_path.is_file():
            target_img = Image.open(str(target_img_path))
            target_rgb_arr = np.asarray(target_img)
            source_rgb_arr = target_rgb_arr
            # https://stackoverflow.com/a/40056251/2334951
            # out = [rgb for rgb in in_rgb_arr if rgb not in source_rgb_arr[0]]
            A, B = in_rgb_arr, source_rgb_arr[0]
            dims = np.maximum(B.max(0),A.max(0), dtype="uint16")+1
            keep = ~np.in1d(np.ravel_multi_index(A.T,dims),np.ravel_multi_index(B.T,dims))
            #out = A[keep]
            in_rgb_arr = in_rgb_arr[keep]
            in_lab_arr = in_lab_arr[keep]
        else:
            print(f'=== LEVEL {level} ===')
            cfails = 0
            source_hue_arr = colour.RGB_to_HSV(source_rgb_arr / 255)[0][:,0]
            source_lab_arr = rgbarr_to_labarr(source_rgb_arr)
            #print(source_rgb_arr.shape, source_lab_arr.shape, source_hue_arr.shape)
            target_rgb_arr = np.copy(source_rgb_arr)
            new_rgbs = set()
            for c in range(len(source_hue_arr)):
                hue_range0 = source_hue_arr[c], source_hue_arr[(c+1) % source_hue_arr.shape[0]]
                if hue_range0[0] <= hue_range0[1]:
                    ob = False
                    ii = np.where(np.logical_and(hue_range0[0] <= in_hue_arr, in_hue_arr <= hue_range0[1]))[0]
                    n = ii.shape[0]
                    hue_hue_arr = in_hue_arr[ii]
                    hue_sorted_ii = ii[np.argsort(hue_hue_arr)]
                else:
                    ob = True
                    #ii = np.where(np.logical_or(hue_range0[0] <= in_hue_arr, in_hue_arr <= hue_range0[1]))[0]
                    ii0 = np.where(hue_range0[0] <= in_hue_arr)[0]
                    ii1 = np.where(in_hue_arr <= hue_range0[1])[0]
                    ii = np.hstack((ii0, ii1))
                    n = ii0.shape[0] + ii1.shape[0]
                    hue_hue_arr0 = in_hue_arr[ii0]
                    hue_sorted_ii0 = ii0[np.argsort(hue_hue_arr0)]
                    hue_hue_arr1 = in_hue_arr[ii1]
                    hue_sorted_ii1 = ii1[np.argsort(hue_hue_arr1)]
                    hue_hue_arr = np.hstack((hue_hue_arr0, hue_hue_arr1))
                    hue_sorted_ii = np.hstack((hue_sorted_ii0, hue_sorted_ii1))
                jj = [None] * 2
                mm = [None] * 2
                if n == 0:
                    continue
                elif n == 1:
                    jj[0] = hue_sorted_ii[:1]
                    jj[1] = None
                elif n in {2, 3}:
                    jj[0] = hue_sorted_ii[:1]
                    jj[1] = hue_sorted_ii[-1:]
                elif n in {4, 5}:
                    jj[0] = hue_sorted_ii[:2]
                    jj[1] = hue_sorted_ii[-2:]
                else:
                    jj_slice_size = n//6
                    jj[0] = hue_sorted_ii[jj_slice_size:2*jj_slice_size]
                    jj[1] = hue_sorted_ii[-2*jj_slice_size:-jj_slice_size]

                #if ob:
                #    for i in range(2):
                #        if jj[i] is None: continue
                #        kk0 = jj[i][np.where(hue_range0[0] <= in_hue_arr[jj[i]])[0]]
                #        kk1 = jj[i][np.where(in_hue_arr[jj[i]] <= hue_range0[1])[0]]
                #        if kk0.shape[0] and kk1.shape[0]:
                #            hue_range1[i] = np.min(in_hue_arr[kk0]), np.max(in_hue_arr[kk1])
                #            mm[i] = np.where(np.logical_or(hue_range1[i][0] <= in_hue_arr, in_hue_arr <= hue_range1[i][1]))[0]
                #        elif not kk1.shape[0]:
                #            hue_range1[i] = np.min(in_hue_arr[kk0]), np.max(in_hue_arr[kk0])
                #            mm[i] = np.where(np.logical_and(hue_range1[i][0] <= in_hue_arr, in_hue_arr <= hue_range1[i][1]))[0]
                #        else:
                #            hue_range1[i] = np.min(in_hue_arr[kk1]), np.max(in_hue_arr[kk1])
                #            mm[i] = np.where(np.logical_and(hue_range1[i][0] <= in_hue_arr, in_hue_arr <= hue_range1[i][1]))[0]
                #else:
                #    for i in range(2):
                #        if jj[i] is None: continue
                #        hue_range1[i] = np.min(in_hue_arr[jj[i]]), np.max(in_hue_arr[jj[i]])
                #        mm[i] = np.where(np.logical_and(hue_range1[i][0] <= in_hue_arr, in_hue_arr <= hue_range1[i][1]))[0]

                hue_range1 = [None] * 2
                for i in range(2):
                    if jj[i] is None: continue
                    hue_range1[i] = np.min(in_hue_arr[jj[i]]), np.max(in_hue_arr[jj[i]])

                mm = jj

                #print(c+1, hue_range0, ii.shape[0], flush=True)
                hue_hue_arr = [None] * 2
                hue_rgb_arr = [None] * 2
                hue_lab_arr = [None] * 2
                dE_arr = [None] * 2
                for i in range(2):
                    if jj[i] is None: continue
                    n = mm[i].shape[0]
                    #print(c+1, i, hue_range1[i], n, flush=True)
                    hue_hue_arr[i] = in_hue_arr[mm[i]]
                    hue_rgb_arr[i] = in_rgb_arr[mm[i]]
                    hue_lab_arr[i] = in_lab_arr[mm[i]]
                    dE_arr[i] = np.zeros((len(hue_lab_arr[i])), dtype="float64")
                for i in range(2):
                    if jj[i] is None: continue
                    #print(source_rgb_arr[0][(c + i) % source_rgb_arr.shape[1]], source_hue_arr[(c + i) % source_hue_arr.shape[0]], round(source_hue_arr[(c + i) % source_hue_arr.shape[0]] * 360))
                    j = (c + i) % source_lab_arr.shape[1]
                    color_lab_arr = np.tile(source_lab_arr[0][j], len(hue_lab_arr[i])).reshape((len(hue_lab_arr[i]), 3))
                    dE_arr[i] = color_dE_arr = de2000.delta_e_from_lab(hue_lab_arr[i], color_lab_arr)
                #raise Exception
                value = [None] * 2
                main_rgb = [None] * 2
                pick_rgb = [None] * 2
                for i in range(2):
                    if jj[i] is None: continue
                    for ii in np.argsort(dE_arr[i]):
                        pick_rgb[i] = hue_rgb_arr[i][ii]
                        t_pick_rgb = tuple(pick_rgb[i])
                        if t_pick_rgb in new_rgbs:
                            #print(t_pick_rgb, flush=True)
                            pick_rgb[i] = None
                            continue
                        else:
                            new_rgbs.add(t_pick_rgb)
                            j = (c + i) % source_lab_arr.shape[1]
                            main_rgb[i] = source_rgb_arr[0][j]
                            value[i] = dE_arr[i][ii]
                            break
                #raise Exception
                for i in range(2):
                    if pick_rgb[i] is None: continue
                    main_i = np.where(np.all(target_rgb_arr.reshape((-1, 3)) == main_rgb[i], axis=1))[0][0]
                    pick_i = main_i + (1 - i)
                    target_rgb_arr = np.insert(target_rgb_arr, pick_i, pick_rgb[i], axis=1)

            target_img = Image.fromarray(target_rgb_arr, 'RGB')
            target_img.save(str(target_img_path))

            source_rgb_arr = target_rgb_arr

            # https://stackoverflow.com/a/40056251/2334951
            # out = [rgb for rgb in in_rgb_arr if rgb not in source_rgb_arr[0]]
            A, B = in_rgb_arr, source_rgb_arr[0]
            dims = np.maximum(B.max(0),A.max(0), dtype="uint16")+1
            keep = ~np.in1d(np.ravel_multi_index(A.T,dims),np.ravel_multi_index(B.T,dims))
            #out = A[keep]
            in_rgb_arr = in_rgb_arr[keep]
            in_lab_arr = in_lab_arr[keep]

            #input("PRESS ENTER TO CONTINUE")
