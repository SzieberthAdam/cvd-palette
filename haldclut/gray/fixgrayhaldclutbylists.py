# 3rd party libraries
from PIL import Image
import numpy as np

# standard libraries
import pathlib
import sys

if __name__ == "__main__":

    _thisdir = pathlib.Path(__file__).parent.resolve()
    _importdir = _thisdir.parent.parent.resolve()
    sys.path.insert(0, str(_importdir))

    # 3rd party libraries (module file available)
    import clut

    sys.path.remove(str(_importdir))

    img0path = pathlib.Path(sys.argv[1])
    img1path = img0path.parent / f'{img0path.stem}.fixed{img0path.suffix}'
    clut = clut.CLUT(str(img0path))

    for arg in sys.argv[2:]:
        listpath = pathlib.Path(arg)
        print(listpath)
        with listpath.open("r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                rgb, *_, v = line.split()
                rgb = tuple(int(s) for s in rgb.split(","))
                r, g, b = rgb
                v = int(v)
                #print(f'{rgb} -> {v}')
                clut.clut[r][g][b] = np.array([v, v, v], dtype=np.uint8)

    clut.save(str(img1path),size=16)
