import pathlib
import sys

_thisdir = pathlib.Path(__file__).parent.resolve()
_importdir = _thisdir / "tsarjak"
sys.path.insert(0, str(_importdir))

from tsarjak.recolor import Core

sys.path.remove(str(_importdir))

cvd_map = {
    "deuta": "deutranopia", # SIC
    "prota": "protanopia",
    "trita": "tritanopia",
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

    print(f'"{cb}.tsarjak.png" ...')
    Core.simulate(
        input_path=str(p_identity),
        return_type='save',
        save_path=f'{cb}.tsarjak.png',
        simulate_type=cvd_map[cb],
        simulate_degree_primary=1.0,
    )
    print("done.")
