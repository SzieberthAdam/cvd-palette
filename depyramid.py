import itertools
import pathlib
import sys

from PIL import Image
import numpy as np

def get_boundary_colors(isoluminant_level0_image_path):
    p = pathlib.Path(isoluminant_level0_image_path)
    img = Image.open(str(p))
    arr = np.array(img).reshape((-1, 3))
    print(arr)


if __name__ == "__main__":

    try:
        graylevel = int(sys.argv[1], 16)
        assert graylevel in range(0,256)
    except (IndexError, ValueError, AssertionError):
        print('Usage: python generate.py <graylevel>')
        print('0 <= graylevel <= FF (base 16)')
        sys.exit(1)

    root = pathlib.Path(__file__).parent.resolve()
    imgpath = root.parent / "isoluminant" / "level0" / f'{graylevel:0>2X}.png'
    get_boundary_colors(imgpath)
