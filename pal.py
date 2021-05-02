

def load(fp, *, verify_header=True):
    fp.seek(0, 0)
    s = fp.read().rstrip("\n")
    return loads(s, verify_header=verify_header)

def loads(s, *, verify_header=True):
    l = s.split("\n")
    if verify_header:
        if l[0] != "JASC-PAL":
            raise PaletteError("Invalid JASC Palette first line header. JASC-PAL is expected.")
        if l[1] != "0100":
            raise PaletteError("Invalid JASC Palette second line magic number. 0100 is expected.")
    try:
        num_colors = int(l[2])
    except ValueError:
        raise PaletteError("Invalid JASC Palette third line number of colors. Integer value is expected.")
    linenr = 4
    result = []
    for n in range(num_colors):
        rgbstr = l[linenr-1]
        rgbstrvals = rgbstr.split()
        try:
            rgbvals = [int(s) for s in rgbstrvals]
        except ValueError:
            raise PaletteError(
                f'Invalid JASC Palette {linenr}th line RGB string.\n'
                "Three integers are expected separated by spaces."
            )
        result.append(rgbvals)
        linenr += 1
    return result

def dump(data, fp):
    return fp.write(dumps(data))

def dumps(data):
    lines = ["JASC-PAL", "0100", f'{len(data)}']
    lines += [f'{int(r)} {int(g)} {int(b)}' for r, g, b in data] + [""]
    return "\n".join(lines)


