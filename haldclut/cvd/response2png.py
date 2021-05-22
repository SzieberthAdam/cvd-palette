# standard libraries
import base64
import pathlib
import sys

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f'ERROR! A text file with response content was expected.')
        sys.exit(1)

    pin = pathlib.Path(sys.argv[1])
    with pin.open("rb") as f:
        data = f.read()

    commai = data.find(b',')
    assert 0 < commai, 'Invalid response content.'

    headerdata = data[:commai]
    assert headerdata == b"data:image/png;base64", 'Invalid response content.'

    pngdata = data[commai+1:].rstrip(b"\n ")

    bdata = base64.decodebytes(pngdata)

    pout = pin.parent / f'{pin.stem}.png'
    with pout.open("wb") as f:
        f.write(bdata)
    print(f'"{pout}" written.')

