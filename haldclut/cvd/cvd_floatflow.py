print("https://gitlab.com/FloatFlow/colorblind/-/tree/master/")
print()

import numpy as np
import cv2
from colorblind import colorblind
import matplotlib.pyplot as plt

# load image
img = cv2.imread("..\identity\identity.png")
img = img[..., ::-1]


print('"deuta.floatflow.png" ...')
deuta_img = colorblind.simulate_colorblindness(np.copy(img), colorblind_type='deuteranopia')
cv2.imwrite("deuta.floatflow.png", deuta_img)
print("done.")

print('"prota.floatflow.png" ...')
prota_img = colorblind.simulate_colorblindness(np.copy(img), colorblind_type='protanopia')
cv2.imwrite("prota.floatflow.png", prota_img)
print("done.")

print('"trita.floatflow.png" ...')
trita_img = colorblind.simulate_colorblindness(np.copy(img), colorblind_type='tritanopia')
cv2.imwrite("trita.floatflow.png", trita_img)
print("done.")
