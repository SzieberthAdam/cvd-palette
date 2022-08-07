import decimal
import itertools
import math
import pathlib
import re
import sys

_thisdir = pathlib.Path(__file__).parent.resolve()
_importdir = _thisdir.parent.resolve()
sys.path.insert(0, str(_importdir))

# 3rd party libraries (module file available)
import de2000

sys.path.remove(str(_importdir))

# 3rd party libraries
import colour
from PIL import Image
import numpy as np


def rgbstr(rgb):
    r, g, b = rgb
    return f'#{r:0>2x}{g:0>2x}{b:0>2x}'


rate_ss = ["01x", "08x", "032x"]

HEADER = """
════{hfill11}═╤═{hfill12}═╤════════╤══════════════════════
Name{hfill21} │ {hfill22} │ dE2000 │ Download             
────{hfill31}─┼─{hfill32}─┼────────┼──────────────────────
"""

FOOTER = """
════{ffill11}═╧═{ffill12}═╧════════╧══════════════════════
"""

BODY = "{body1} │ {body2} │ {body3} │ {body4:<21}"

DOWNLOAD_FSTR = '<a href="{hexurl}">HEX</a> <a href="{palurl}">PAL</a> PNG-<a href="{png1xurl}">1x</a>-<a href="{png8xurl}">8x</a>-<a href="{png32xurl}">32x</a>'

visual_fstr = '<span style="color:{rgb_s}">█</span>'

dE_fstr = '<a href="{dEurl}">{dEval}</a>'

if __name__ == "__main__":
    for p in _thisdir.iterdir():
        if not p.is_dir():
            continue

        df = {
            "basename": p.name.upper(),
        }

        img_filename = p.with_suffix(".01x.png").name
        p_img = p / img_filename
        img = Image.open(str(p_img))
        img_arr = np.array(img)
        n_pals, n_colors, _ = img_arr.shape

        txt_filename = p.with_suffix(".txt").name
        p_txt = p / txt_filename

        with p_txt.open("r", encoding="utf8") as _f:
            s_txt = _f.read()
        li_s_dE = re.findall('lowest --\s+(\d+\.\d+)', s_txt)
        li_dE = [decimal.Decimal(s) for s in li_s_dE]

        name_fstr = '{basename}-{pal_n}'
        name_len = max(len(df["basename"]) + 1 + len(str(n_pals)), 4)

        df["hfill11"] = "═" * (name_len - 4)
        df["hfill12"] = "═" * n_colors
        df["hfill21"] = " " * (name_len - 4)
        df["hfill22"] = "Visual"[:n_colors] + " " * (n_colors - len("Visual"[:n_colors]))
        df["hfill31"] = "─" * (name_len - 4)
        df["hfill32"] = "─" * n_colors
        df["ffill11"] = "═" * (name_len - 4)
        df["ffill12"] = "═" * n_colors

        header = HEADER.strip().format(**df)
        footer = FOOTER.strip().format(**df)


        names = []
        for pal_n in range(1, n_pals+1):
            df["pal_n"] = pal_n
            name = name_fstr.format(**df)
            names.append(name)
        del df["pal_n"]

        visuals = []
        for i in range(n_pals):
            visual = ""
            pal_n = i + 1
            pal_arr = img_arr[i]
            for rgb in pal_arr:
                df["rgb_s"] = rgbstr(rgb).upper()
                visual += visual_fstr.format(**df)
            del df["rgb_s"]
            visuals.append(visual)

        downloads = []
        for i in range(n_pals):
            pal_n = i + 1
            base_filepath = pathlib.Path(f'{df["basename"]}-{pal_n:0>4}')
            df["hexurl"] = base_filepath.with_suffix(".hex")
            df["palurl"] = base_filepath.with_suffix(".pal")
            df["png1xurl"] = base_filepath.with_suffix(".01x.png")
            df["png8xurl"] = base_filepath.with_suffix(".08x.png")
            df["png32xurl"] = base_filepath.with_suffix(".32x.png")
            download = DOWNLOAD_FSTR.format(**df)
            downloads.append(download)

        parts = [header]
        for i in range(n_pals):
            pal_n = i + 1
            base_filepath = pathlib.Path(f'{df["basename"]}-{pal_n:0>4}')
            df["body1"] = names[i] + " " * (name_len - len(names[i]))
            df["body2"] = visuals[i]
            df["dEurl"] = base_filepath.with_suffix(".txt")
            df["dEval"] = f'{li_dE[i]:.3f}'
            df["body3"] = " " * (6 - len(df["dEval"])) + dE_fstr.format(**df)
            df["body4"] = downloads[i]
            body = BODY.format(**df)
            parts.append(body)


        parts.append(footer)
        result_s = "\n".join(parts)

        template_filename = pathlib.Path("template.htm")
        with template_filename.open("r", encoding="utf8") as f:
            template = f.read()

        out_filename = p.with_suffix(".table.htm")
        with out_filename.open("w", encoding="utf8", newline='\r\n') as f:
            f.write(template.replace("{{table}}",result_s))
        print(out_filename)
