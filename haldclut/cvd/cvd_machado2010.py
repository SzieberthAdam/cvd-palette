from PIL import Image
import numpy as np

# https://gitlab.com/brohrer/lodgepole
# from lodgepole.image_tools import gamma_decompress
def gamma_decompress(img):
    """
    Make pixel values perceptually linear.
    """
    img_lin = ((img + 0.055) / 1.055) ** 2.4
    i_low = np.where(img <= .04045)
    img_lin[i_low] = img[i_low] / 12.92
    return img_lin

# https://gitlab.com/brohrer/lodgepole
# from lodgepole.image_tools import gamma_compress
def gamma_compress(img_lin):
    """
    Make pixel values display-ready.
    """
    img = 1.055 * img_lin ** (1 / 2.4) - 0.055
    i_low = np.where(img_lin <= .0031308)
    img[i_low] = 12.92 * img_lin[i_low]
    return img

cvd_mx = {
    "deuta": np.array([
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ]),

    "prota": np.array([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ]),

    "trita": np.array([
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900],
    ]),

}


identity_img = Image.open("..\identity\identity.png")

arr = gamma_decompress(np.array(identity_img, dtype="float64") / 255)

for cvd, mx in cvd_mx.items():
    a = np.dot(arr.reshape((-1, 3)), mx.T).reshape((4096, 4096, 3))
    b = np.minimum(np.maximum(a, 0), 1)
    c = gamma_compress(b)
    d = np.minimum(np.maximum(c, 0), 1)
    arr_grayscale = (d * 255).round().astype("uint8")
    img_grayscale = Image.fromarray(arr_grayscale, 'RGB')
    img_grayscale.save(f'{cvd}.machado2010.png')
    print(f'"{cvd}.machado2010.png" saved.')
    #break
