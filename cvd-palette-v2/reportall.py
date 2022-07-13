# standard libraries
import pathlib
import sys

_thisdir = pathlib.Path(__file__).parent.resolve()
_importdir = _thisdir.parent.resolve()
sys.path.insert(0, str(_importdir))

sys.path.remove(str(_importdir))

# 3rd party libraries
from PIL import Image
import numpy as np

import sortall

if __name__ == "__main__":

    np.seterr(all='raise')

    root = pathlib.Path(__file__).parent.resolve()

    usage = """Usage: python reportall.py <image of palettes>"""

    try:
        img_path = pathlib.Path(sys.argv[1])
    except (IndexError, ValueError, FileNotFoundError):
        print(usage)
        sys.exit(1)

    img = Image.open(str(img_path))
    rgb_arr = np.array(img)

    report_str = sortall.report_str(rgb_arr)
    report_str_path = img_path.with_suffix(".txt")
    if report_str_path.is_file():
        report_str_bak_path = report_str_path.with_suffix(".bak" + report_str_path.suffix)
        report_str_bak_path.unlink(missing_ok=True)
        report_str_path.rename(report_str_bak_path)
    with report_str_path.open("w", encoding="utf8", newline='\r\n') as f:
        f.write(report_str)
