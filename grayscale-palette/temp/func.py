from generate import *
def do():
  for i in range(n):
    x = 255*i/(n-1)
    if x==int(x):
      print(f'{int(x):>2X} -> {guess_y(n, i):>2X}')
