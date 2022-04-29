# 3rd party libraries
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

# standard libraries
import collections
from functools import partial
import pathlib
import random
import sys

_thisdir = pathlib.Path(__file__).parent.resolve()
_importdir = _thisdir.parent.parent.resolve()
sys.path.insert(0, str(_importdir))

# 3rd party libraries (module file available)
import clut

sys.path.remove(str(_importdir))



level = 4
randomRGBn = 100

dist = int(256 / level)
refpoints = tuple(range(0, 257, dist))
width, height = 80, 15

def iterrgb(points=None):
    points = points or tuple(range(256))
    for r in points:
        for g in points:
            for b in points:
                yield r, g, b

grayclut_paths = list(_thisdir.glob("gray.*.png"))
grayclut = {}
for p in grayclut_paths:
    print(f'generating hald clut for {p} ...')
    grayclut[p.stem[5:]] = clut.CLUT(str(p))

allpoints = list(iterrgb(refpoints))
for _ in range(randomRGBn):
    R, G, B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    allpoints.append((R, G, B))

img = Image.new('RGB', (width*len(grayclut), height*(len(allpoints))))

drawctx = ImageDraw.Draw(img)  # draw context

for c, name in enumerate(sorted(grayclut)):
    print(name)
    eh = grayclut[name]
    for r, RGB in enumerate(allpoints):
        RGB = tuple(max(0, min(255, x)) for x in RGB)
        x0, y0 = width*c,     height*r
        x1, y1 = width*(c+1), height*(r+1)
        drawctx.rectangle([x0, y0, x1, y1], fill=RGB)
        text = name
        if RGB == (0, 0, 0):
            gray_RGB = (255, 255, 255)
        elif RGB == (255, 255, 255):
            gray_RGB = (0, 0, 0)
        else:
            gray_RGB = tuple(eh.clut[RGB[0]][RGB[1]][RGB[2]])
            text = "0123456789"
        w, h = drawctx.textsize(text)
        drawctx.text([x0+(width-w)//2, y0+2], text, fill=gray_RGB)

img.save("compare.png")
