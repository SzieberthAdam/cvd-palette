from skimage import color
from skimage import io

img = color.rgb2gray(io.imread("..\identity\identity.png"))


io.imsave('gray.skimage.png', img)
