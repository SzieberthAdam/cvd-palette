import os
import pathlib
import sys

from PIL import Image

workdir = pathlib.Path(os.getcwd()).resolve()

parts = {
    "ul": (    0,    0),
    "ur": ( 2048,    0),
    "dl": (    0, 2048),
    "dr": ( 2048, 2048),
}

if __name__ == "__main__":

    for p in workdir.glob("*.ul.png"):
        img_parts = {"ul": Image.open(str(p))}
        for part in set(parts) - {"ul"}:
            pp = p.parent / f'{p.stem[:-2]}{part}.png'
            if pp.is_file():
                img_parts[part] = Image.open(str(pp))
        if len(img_parts) < 4:
            print(f'MISSING PARTS! "{p}"')
            continue

        filename = f'{p.stem[:-3]}.png'

        print(f'"{filename}" ...')

        img = Image.new("RGB", (4096, 4096), "white")

        for part, part_img in img_parts.items():
            part_xy = parts[part]
            img.paste(part_img, part_xy)

        img.save(filename)
        print("done.")
