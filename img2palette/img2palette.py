# standard libraries
from functools import partial
import collections
import math
import pathlib
import random
import statistics
import sys

# 3rd party libraries
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import numpy as np

_thisdir = pathlib.Path(__file__).parent.resolve()
_importdir = _thisdir.parent.resolve()
sys.path.insert(0, str(_importdir))

# 3rd party libraries (module file available)
import clut

sys.path.remove(str(_importdir))

def rgbstr(rgb):
    r, g, b = rgb
    return f'#{r:0>2X}{g:0>2X}{b:0>2X}'


haldclut_path = _thisdir / "img2pal4096.png"
haldclut_path = pathlib.Path(sys.argv[1])
haldclut = clut.CLUT(str(haldclut_path))

in_img_path = pathlib.Path(sys.argv[2])
in_img = Image.open(str(in_img_path))
in_arr = np.array(in_img, dtype="uint8")

clut_arr = haldclut(in_arr)
clut_img = Image.fromarray(clut_arr, 'RGB')
clut_img.save(f'{in_img_path.stem}-{haldclut_path.stem}.png')

c = collections.Counter(tuple(a) for a in clut_arr.reshape((-1, 3)))
median = statistics.median(c.values())
stdev = statistics.pstdev(c.values())

L = sorted(c, key=c.get, reverse=True)

horiz_items = 8
vert_items = math.ceil(len(c) / horiz_items)
width, height = 100, 15

out_img = Image.new('RGB', (horiz_items*width, vert_items*height))
drawctx = ImageDraw.Draw(out_img)  # draw context
for i, rgb in enumerate(L):
    row, col = divmod(i, horiz_items)
    x0, y0 = width*col,     height*row
    x1, y1 = width*(col+1) - 1, height*(row+1) - 1
    drawctx.rectangle([x0, y0, x1, y1], fill=rgb)
    text = f'{rgbstr(rgb)} {c[rgb]}'
    w, h = drawctx.textsize(text)
    text_rgb = ((0, 0, 0) if 128 < statistics.mean(rgb) else (255, 255, 255))
    drawctx.text([x0+(width-w)//2, y0+2], text, fill=text_rgb)

out_img.save(f'{in_img_path.stem}-{haldclut_path.stem}-colors.png')
