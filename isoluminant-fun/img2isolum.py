# 3rd party libraries
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import numpy as np

# standard libraries
import pathlib
import sys

_thisdir = pathlib.Path(__file__).parent.resolve()
_importdir = _thisdir.parent.resolve()
sys.path.insert(0, str(_importdir))

# 3rd party libraries (module file available)
import clut

sys.path.remove(str(_importdir))


grayclut_path = _thisdir.parent / "haldclut" / "gray" / "gray.szieberth.png"
#grayclut_path = _thisdir.parent / "haldclut" / "gray" / "gray.gimp2-10.png"
#grayclut_path = _thisdir.parent / "haldclut" / "gray" / "gray.ps14.png"
isoluminant_path = _thisdir.parent / "isoluminant" / "szieberth"
#isoluminant_path = _thisdir.parent / "isoluminant" / "gimp2-10"
#isoluminant_path = _thisdir.parent / "isoluminant" / "ps14"


if __name__ == "__main__":

    usagestr = 'Usage: python img2isolum.py <image>'

    try:
        in_img_path = pathlib.Path(sys.argv[1])
    except (IndexError, ValueError):
        print(usagestr)
        sys.exit(1)
    if not in_img_path.is_file():
        print(usagestr)
        sys.exit(1)

    in_img = Image.open(str(in_img_path))
    grayclut = clut.CLUT(str(grayclut_path))

    iso_arrs = [None for _ in range(256)]
    for v in range(256):
        iso_img_path = isoluminant_path / f'{v:0>2x}.png'
        iso_img = Image.open(str(iso_img_path))
        iso_arr = np.array(iso_img).reshape((-1, 3))
        iso_arrs[v] = iso_arr


    gray_img_arr = grayclut(in_img) # shape: (w, h, 3)
    gray_img = Image.fromarray(gray_img_arr, 'RGB')
    gray_img_path = in_img_path.with_suffix(".gray.png")
    gray_img.save(str(gray_img_path))

    gray_arr = gray_img_arr[:, :, 0] # shape: (w, h)

    rng = np.random.default_rng()

    out_img_arr = np.array([rng.choice(iso_arrs[v]) for v in gray_arr.reshape((-1,))]).reshape(gray_img_arr.shape)

    out_img = Image.fromarray(out_img_arr, 'RGB')
    out_img_path = in_img_path.with_suffix(".isolum.png")
    out_img.save(str(out_img_path))
