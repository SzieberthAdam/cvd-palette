from PIL import Image
img = Image.open("..\identity\identity.png").convert("LA")
img.save("gray.pil.png")
