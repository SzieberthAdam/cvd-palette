# https://www.kdnuggets.com/2019/12/convert-rgb-image-grayscale.html
# https://gitlab.com/brohrer/lodgepole
import lodgepole.image_tools as lit

from PIL import Image
import numpy as np

img = np.asarray(Image.open("..\identity\identity.png")) / 255

grayscale = lit.rgb2gray_approx(img) * 255
arr_grayscale = np.ndarray.round(grayscale).astype("uint8").repeat(3).reshape((4096, 4096, 3))
img_grayscale = Image.fromarray(arr_grayscale, 'RGB')
img_grayscale.save("gray.lodgepole.approx.png")

grayscale = lit.rgb2gray(img) * 255
arr_grayscale = np.ndarray.round(grayscale).astype("uint8").repeat(3).reshape((4096, 4096, 3))
img_grayscale = Image.fromarray(arr_grayscale, 'RGB')
img_grayscale.save("gray.lodgepole.png")

#gray_img = lit.rgb2gray(img)
