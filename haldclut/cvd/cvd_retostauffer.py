import pathlib
import sys

from PIL import Image
from colorspace import cvd_emulator

_thisdir = pathlib.Path(__file__).parent.resolve()

cvd_map = {
    "deuta": "deutan",
    "prota": "protan",
    "trita": "tritan",
}

parts = {
    "ul": (    0,    0),
    "ur": ( 2048,    0),
    "dl": (    0, 2048),
    "dr": ( 2048, 2048),
}

p_identity = {
    part: (_thisdir.parent / "identity" / f'identity.{part}.png').resolve()
    for part in parts
}

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f'ERROR! CVD type (deuta, prota, trita) must be given as argument.')
        sys.exit(1)
    else:
        cb = sys.argv[1].lower()

    if cb not in cvd_map:
        print(f'ERROR! CVD type (deuta, prota, trita) must be given as argument.')
        sys.exit(1)

    filename = f'{cb}.retostauffer.png'

    print(f'"{filename}" ...')
    img = Image.new("RGB", (4096, 4096), "white")
    for part, part_xy in parts.items():

        part_filename = f'{cb}.retostauffer.{part}.png'
        part_filepath = (_thisdir / part_filename).resolve()

        cvd_emulator(
            image=str(p_identity[part]),
            cvd=cvd_map[cb],
            severity=1,
            output=part_filename,
            dropalpha=True,
        )

        part_img = Image.open(part_filename)
        img.paste(part_img, part_xy)
        part_filepath.unlink()
    img.save(filename)
    print("done.")
