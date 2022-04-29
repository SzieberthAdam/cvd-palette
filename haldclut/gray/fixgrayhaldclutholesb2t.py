# 3rd party libraries
from PIL import Image
import numpy as np

# standard libraries
import pathlib
import sys

def iterrgb(points=None):
    points = points or tuple(range(256))
    for r in points:
        for g in points:
            for b in points:
                yield r, g, b

def iterhigherpoints(r, g, b):
    s = set()
    s.add((r, g, min(b+1, 255)))
    s.add((r, min(g+1, 255), b))
    s.add((r, min(g+1, 255), min(b+1, 255)))
    s.add((min(r+1, 255), g, b))
    s.add((min(r+1, 255), g, min(b+1, 255)))
    s.add((min(r+1, 255), min(g+1, 255), min(b+1, 255)))
    s -= {(r,g,b),}
    yield from iter(s)

if __name__ == "__main__":

    _thisdir = pathlib.Path(__file__).parent.resolve()
    _importdir = _thisdir.parent.parent.resolve()
    sys.path.insert(0, str(_importdir))

    # 3rd party libraries (module file available)
    import clut

    sys.path.remove(str(_importdir))

    img0path = pathlib.Path(sys.argv[1])
    img1path = img0path.parent / f'{img0path.stem}.fixedholesb2t{img0path.suffix}'
    clut = clut.CLUT(str(img0path))

    log = []

    for pr0, pg0, pb0 in iterrgb(tuple(range(255,-1,-1))):
        #print()
        v0 = clut.clut[pr0][pg0][pb0]
        assert len(set(v0)) == 1  # gray
        v = v0[0]
        #print(f'{pr0},{pg0},{pb0} : {v0}')
        v2 = v
        for pr1, pg1, pb1 in iterhigherpoints(pr0, pg0, pb0):
            v1 = clut.clut[pr1][pg1][pb1]
            assert len(set(v1)) == 1  # gray
            #print(f'{pr1},{pg1},{pb1} : {v1}')
            v2 = min(v2, v1[0])
        if v != v2:
            log.append(f'{pr0},{pg0},{pb0} : {v} -> {v2}')
            print(log[-1])
        clut.clut[pr0][pg0][pb0] = np.array([v2, v2, v2], dtype=np.uint8)

    clut.save(str(img1path),size=16)
    with open(str(img1path) + ".txt", "w", encoding="utf8") as f:
        f.write("\n".join(log))
