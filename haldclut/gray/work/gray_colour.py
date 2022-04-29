#https://github.com/colour-science/colour/
import colour
from PIL import Image
import numpy as np

img = Image.open("..\identity\identity.png")
rgb_arr = np.array(img)

y_arr = colour.RGB_luminance(rgb_arr, )

#xyz_arr = colour.sRGB_to_XYZ(rgb_arr/255)
#
#y_arr = xyz_arr[:, :,1]
#arr_grayscale = np.ndarray.round(y_arr*255).astype("uint8").repeat(3).reshape((4096, 4096, 3))
#img_grayscale = Image.fromarray(arr_grayscale, 'RGB')
#img_grayscale.save("gray.colour.y.png")
#
#lab_arr = colour.XYZ_to_Lab(xyz_arr)
#
#l_arr = lab_arr[:, :,0]
#arr_grayscale = np.ndarray.round(l_arr*255/100).astype("uint8").repeat(3).reshape((4096, 4096, 3))
#img_grayscale = Image.fromarray(arr_grayscale, 'RGB')
#img_grayscale.save("gray.colour.l.png")
