# https://www.kdnuggets.com/2019/12/convert-rgb-image-grayscale.html
# https://gitlab.com/brohrer/lodgepole
import lodgepole.image_tools as lit

from PIL import Image
import numpy as np

from skimage import color
from skimage import io


def rgb2gray_linear(rgb_img):
    """
    Convert *linear* RGB values to *linear* grayscale values.
    """
    red = rgb_img[:, :, 0]
    green = rgb_img[:, :, 1]
    blue = rgb_img[:, :, 2]

    gray_img = np.sqrt(
        (0.299 * red)**2
        + (0.587 * green)**2
        + (0.114 * blue)**2
    )

    return gray_img


img = io.imread("..\identity\identity.png")

#hsv_arr = color.rgb2hsv(img)
