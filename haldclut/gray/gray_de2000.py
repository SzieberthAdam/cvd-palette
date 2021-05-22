import pathlib
import shutil
import sys
import datetime

from PIL import Image
import numpy as np

_thisdir = pathlib.Path(__file__).parent.resolve()
_importdir = _thisdir.parent.parent.resolve()
sys.path.insert(0, str(_importdir))

# 3rd party libraries (module file available)
import de2000
import clut

sys.path.remove(str(_importdir))

p_de2000_img = (_thisdir / "gray.de2000.png").resolve()
if not p_de2000_img.is_file():
    p_identity_img = (_thisdir / "../identity/identity.png").resolve()
    shutil.copy(p_identity_img, p_de2000_img)

img = Image.open(str(p_de2000_img))
arr = np.array(img).reshape((-1, 3))

gray_arr = np.array(range(256), dtype="uint8").repeat(3).reshape((-1,3))
gray_lab_arr = de2000.get_lab_arr(gray_arr)

print("WARNING! Long process. There will be 4096 timestamped messages...")
for i, RGB in enumerate(np.copy(arr)):
    #if i < 321617:                                           # for testing only
    #    continue                                             # for testing only
    sR, sG, sB = RGB
    if sR == sG == sB:
        continue
    if not (i % 4096):
        print(f'{datetime.datetime.now()}  row: {1+(i//4096)}')
        arr_grayscale = arr.reshape((4096, 4096, 3))
        img_grayscale = Image.fromarray(arr_grayscale, 'RGB')
        img_grayscale.save("gray.de2000.png")

    lab1 = np.array([de2000.get_lab_arr(RGB)], dtype="float64")

    mindE = 999999999
    mindEgRGB = None

    dE_arr = de2000.delta_e_from_lab(gray_lab_arr, lab1)
    grayv = np.argmin(dE_arr)
    arr[i] = grayv

    #for j, (gR, gG, gB) in enumerate(gray_arr):
    #    lab2 = gray_lab_arr[j]
    #    dE = de2000.delta_e_from_lab(lab1, lab2)
    #    #print((gR, gG, gB), dE, mindE)                       # for testing only
    #    if dE < mindE:
    #        #print("*")                                       # for testing only
    #        mindE = dE
    #        mindEgRGB = gR, gG, gB
    #arr[i] = mindEgRGB

    #print(f'{RGB} {arr[i]} {mindE}')                         # for testing only
    #break                                                    # for testing only

else:
    arr_grayscale = arr.reshape((4096, 4096, 3))

    img_grayscale = Image.fromarray(arr_grayscale, 'RGB')
    img_grayscale.save("gray.de2000.png")
