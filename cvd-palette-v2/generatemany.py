import pathlib
import sys

from PIL import Image
import numpy as np

import generate1
import sortall


if __name__ == "__main__":

    usage = """Usage: python generatemany.py <palette size or image> <pyramid level1 colors> <next-level-neighbour-colors> <next-level-closest-de-colors> [minimum dE (default 10) [number-of-palettes (default infinite)]] """
    try:
        isopyramid_start_colors = int(sys.argv[2])
        n_neighbour_colors = int(sys.argv[3])
        n_close_dE_colors = int(sys.argv[4])
    except (IndexError, ValueError):
        print(usage)
        sys.exit(1)

    if sys.argv[1].isdecimal():
        n_colors_or_rgb_arr = int(sys.argv[1])
        out_img_path = pathlib.Path(f'generate-{n_colors_or_rgb_arr}.png')
        c = 0
    else:
        in_img_path = pathlib.Path(sys.argv[1])
        in_img = Image.open(str(in_img_path))
        n_colors_or_rgb_arr = np.array(in_img, dtype="uint8")
        out_img_path = in_img_path
        c = n_colors_or_rgb_arr.shape[0]

    if 5 < len(sys.argv):
        try:
            min_nci_dE = float(sys.argv[5])
        except (IndexError, ValueError):
            print(usage)
            sys.exit(1)
    else:
        min_nci_dE = 10.0

    if 6 < len(sys.argv):
        try:
            C = int(sys.argv[6])
        except (IndexError, ValueError):
            print(usage)
            sys.exit(1)
    else:
        C = None


    out_rgb_arr = n_colors_or_rgb_arr

    while (C is None or c < C):
        c += 1
        print(f'### PALETTE {c} ###')

        out_rgb_arr = generate1.main(out_rgb_arr, isopyramid_start_colors, n_neighbour_colors, n_close_dE_colors, min_nci_dE=min_nci_dE)
        out_img = Image.fromarray(out_rgb_arr, 'RGB')

        if out_img_path.is_file():
            out_img_bak_path = out_img_path.with_suffix(".bak" + out_img_path.suffix)
            out_img_bak_path.unlink(missing_ok=True)
            out_img_path.rename(out_img_bak_path)

        out_img.save(out_img_path)

        report_str = sortall.report_str(out_rgb_arr)
        report_str_path = out_img_path.with_suffix(".txt")
        if report_str_path.is_file():
            report_str_bak_path = report_str_path.with_suffix(".bak" + report_str_path.suffix)
            report_str_bak_path.unlink(missing_ok=True)
            report_str_path.rename(report_str_bak_path)
        with report_str_path.open("w", encoding="utf8", newline='\r\n') as f:
            f.write(report_str)
