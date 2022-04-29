import cv2
image = cv2.imread("..\identity\identity.png")
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray.cv2.png", grayscale)
