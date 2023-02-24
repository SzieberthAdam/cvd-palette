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

LEVELS = (0x00, 0x33, 0x66, 0x99, 0xCC, 0xFF)

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
for i in range(256**2):
    print(f'{i} of {256**2-1}')
    A = np.array([next(gen) for _ in range(256 * to_lab_arr.shape[0])])
    A0 = A[:,0]
    A1 = A[:,1]
    dE_arr = de2000.delta_e_from_lab(A0, A1).reshape((-1, to_lab_arr.shape[0]))
    M = np.argmin(dE_arr, 1)
    B = to_arr[M]
    O[(i)*256:(i+1)*256] = B
    if not (i % 16):
        img = Image.fromarray(O.reshape((4096, 4096, 3)), 'RGB')
        img.save(f'out216-{i:0>5}.png')
else:
    img = Image.fromarray(O.reshape((4096, 4096, 3)), 'RGB')
    img.save(f'out216.png')
