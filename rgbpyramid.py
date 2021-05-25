BOTTOMLEVEL = 2
TOPLEVEL = 256

def iterrgb(points=None):
    points = points or tuple(range(256))
    for r in points:
        for g in points:
            for b in points:
                yield r, g, b

def get_ref_rgbs(level, max255=False):
    dist = int(256 / level)
    ref_rgbs = []
    for r in range(0, 257, dist):
        for g in range(0, 257, dist):
            for b in range(0, 257, dist):
                if max255:
                    rgb = (min(r,255), min(g,255), min(b,255))
                else:
                    rgb = r, g, b
                ref_rgbs.append(rgb)
    return tuple(sorted(set(ref_rgbs)))

def get_ref_rgb(rgb, level, max255=False):
    dist = int(256 / level)
    result = [None] * 3
    for i in range(3):
        d, m = divmod(rgb[i], dist)
        v = dist * (d + (1 if dist <= 2*m else 0))
        if max255:
            v = min(v, 255)
        result[i] = v
    return tuple(result)

def get_distance(rgb1, rgb2):
    return (rgb1[0]-rgb2[0])**2 + (rgb1[1]-rgb2[1])**2 + (rgb1[2]-rgb2[2])**2
