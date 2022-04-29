import mahotas as mh
import numpy as np
img = mh.imread("..\identity\identity.png")
grayscale = mh.colors.rgb2gray(img)
uint8_grayscale = np.ndarray.round(grayscale).astype("uint8")
mh.imsave("gray.mahotas.png", uint8_grayscale)
