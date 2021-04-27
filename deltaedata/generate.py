# standard libraries
import itertools

# 3rd party libraries
import colour
import numpy as np


def get_lab_arr(rgb_arr):
    xyz_arr = colour.sRGB_to_XYZ(rgb_arr/255)
    lab_arr = colour.XYZ_to_Lab(xyz_arr)
    return lab_arr


def delta_e_from_lab(lab1, lab2):
    delta_E = colour.difference.delta_E_CIE2000(lab1, lab2)
    return delta_E


if __name__ == "__main__":

    gray_rgb_arr = np.repeat(np.arange(256), 3).reshape((256,3))
    gray_lab_arr = get_lab_arr(gray_rgb_arr)

    mono_arr = np.zeros((256,256), dtype="<f4")  # little endian 32-bit float
    for g1, g2 in itertools.combinations_with_replacement(tuple(range(0, 256)), 2):
        dE = delta_e_from_lab(gray_lab_arr[g1], gray_lab_arr[g2])
        mono_arr[g1, g2] = dE
        if g1 != g2:
            mono_arr[g2, g1] = dE
    mono_arr.tofile("mono_deltae.dat")
    print('"mono_deltae.dat" saved.')
    # load like this:
    # arr = np.fromfile("mono_deltae.dat", dtype="<f4").reshape((256,256))

    rgb_arr = np.stack(np.meshgrid(np.arange(256), np.arange(256), np.arange(256)), -1).reshape(-1, 3)
    lab_arr = get_lab_arr(rgb_arr)

    black_arr = delta_e_from_lab(lab_arr[0], lab_arr).astype("<f4")
    black_arr.tofile("black_rgb_deltae.dat")
    print('"black_rgb_deltae.dat" saved.')

    white_arr = delta_e_from_lab(lab_arr[-1], lab_arr).astype("<f4")
    white_arr.tofile("white_rgb_deltae.dat")
    print('"white_rgb_deltae.dat" saved.')
