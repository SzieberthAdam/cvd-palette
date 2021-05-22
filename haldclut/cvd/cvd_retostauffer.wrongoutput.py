import pathlib
import sys

from colorspace import cvd_emulator

_thisdir = pathlib.Path(__file__).parent.resolve()

cvd_map = {
    "deuta": "deutan",
    "prota": "protan",
    "trita": "tritan",
}

p_identity = (_thisdir.parent / "identity" / "identity.png").resolve()

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f'ERROR! CVD type (deuta, prota, trita) must be given as argument.')
        sys.exit(1)
    else:
        cb = sys.argv[1].lower()

    if cb not in cvd_map:
        print(f'ERROR! CVD type (deuta, prota, trita) must be given as argument.')
        sys.exit(1)

    print(f'"{cb}.retostauffer.png" ...')
    cvd_emulator(
        image=str(p_identity),
        cvd=cvd_map[cb],
        severity=1,
        output=f'{cb}.retostauffer.png',
        dropalpha=True,
    )
    print("done.")
