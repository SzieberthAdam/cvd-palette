import pathlib
import shutil
import sys
import datetime
import itertools

from PIL import Image
import numpy as np

_thisdir = pathlib.Path(__file__).parent.resolve()
_importdir = _thisdir.parent.resolve()
sys.path.insert(0, str(_importdir))

# 3rd party libraries (module file available)
import de2000
import clut

sys.path.remove(str(_importdir))

LEVELS = (0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF)


p_identity_img = (_thisdir / "../haldclut/identity/identity.png").resolve()
img = Image.open(str(p_identity_img))
arr = np.array(img).reshape((-1, 3))
print("lab_arr")
lab_arr = de2000.get_lab_arr(arr)

to_arr = np.array(list(itertools.product(LEVELS, LEVELS, LEVELS)), dtype="uint8")
print("to_lab_arr")
to_lab_arr = de2000.get_lab_arr(to_arr)

gen = itertools.product(lab_arr, to_lab_arr)
print("A")
#O = arr
O = np.copy(arr)
startrow = 2764
# startrow = 0
for i in range(startrow*256*16, 256**3):
    rowi = i//256//16
    coli = i - rowi * 256 * 16
    print(i)
    lab = np.tile(lab_arr[i], to_lab_arr.shape[0]).reshape((to_lab_arr.shape[0], 3))
    dE_arr = de2000.delta_e_from_lab(lab, to_lab_arr)
    j = np.argmin(dE_arr)
    rgb = to_arr[j]
    O[i] = rgb
    if coli == 4096 - 1:
        print(f'-> out{rowi:0>4}.png')
        img = Image.fromarray(O.reshape((4096, 4096, 3)), 'RGB')
        img.save(f'out{rowi:0>4}.png')
        #raise Exception
    #if i == 32:
    #    raise Exception


# p_de2000_img = (_thisdir / "gray.de2000.png").resolve()
# if not p_de2000_img.is_file():
#     p_identity_img = (_thisdir / "../identity/identity.png").resolve()
#     shutil.copy(p_identity_img, p_de2000_img)
#
# img = Image.open(str(p_de2000_img))
# arr = np.array(img).reshape((-1, 3))
#
# to_arr = np.array(range(256), dtype="uint8").repeat(3).reshape((-1,3))
# to_lab_arr = de2000.get_lab_arr(to_arr)
#
# print("WARNING! Long process. There will be 4096 timestamped messages...")
# for i, RGB in enumerate(np.copy(arr)):
#     #if i < 321617:                                           # for testing only
#     #    continue                                             # for testing only
#     sR, sG, sB = RGB
#     if sR == sG == sB:
#         continue
#     if not (i % 4096):
#         print(f'{datetime.datetime.now()}  row: {1+(i//4096)}')
#         arr_grayscale = arr.reshape((4096, 4096, 3))
#         img_grayscale = Image.fromarray(arr_grayscale, 'RGB')
#         img_grayscale.save("gray.de2000.png")
#
#     lab1 = np.array([de2000.get_lab_arr(RGB)], dtype="float64")
#
#     mindE = 999999999
#     mindEgRGB = None
#
#     dE_arr = de2000.delta_e_from_lab(to_lab_arr, lab1)
#     grayv = np.argmin(dE_arr)
#     arr[i] = grayv
#
#     #for j, (gR, gG, gB) in enumerate(to_arr):
#     #    lab2 = to_lab_arr[j]
#     #    dE = de2000.delta_e_from_lab(lab1, lab2)
#     #    #print((gR, gG, gB), dE, mindE)                       # for testing only
#     #    if dE < mindE:
#     #        #print("*")                                       # for testing only
#     #        mindE = dE
#     #        mindEgRGB = gR, gG, gB
#     #arr[i] = mindEgRGB
#
#     #print(f'{RGB} {arr[i]} {mindE}')                         # for testing only
#     #break                                                    # for testing only
#
# else:
#     arr_grayscale = arr.reshape((4096, 4096, 3))
#
#     img_grayscale = Image.fromarray(arr_grayscale, 'RGB')
#     img_grayscale.save("gray.de2000.png")
