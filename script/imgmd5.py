from PIL import Image
import numpy as np

import hashlib
import os
import pathlib
import sys

workdir = pathlib.Path(os.getcwd()).resolve()

if __name__ == "__main__":

    if len(sys.argv) < 2:
        ext = "png"
    else:
        ext = sys.argv[1].lstrip(".")

    paths = sorted(workdir.glob(f'*.{ext}'))

    items = []

    for p in paths:
        img = Image.open(str(p))
        arr = np.array(img).flatten()
        m = hashlib.md5(arr)
        md5 = m.hexdigest()
        items.append([md5, p])
        print(f'{md5}  {p.name}')

    items.sort()

    s = "\n".join(f'{md5}  {p.name}' for md5, p in items) + "\n"

    md5p = workdir / f'_{ext}.imgmd5'
    with md5p.open("w") as f:
        f.write(s)

    print(f'"_{md5p}" saved.')
