import cv2
import numpy as np

kernel = np.ones((5, 5), np.uint8)

# === GET THE IMAGE === #
# Read the image and stock it
img = cv2.imread("Dice.jpg", cv2.IMREAD_COLOR)

# === THRESHOLD THE IMAGE === #
# Mise en noir et blanc 
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Seuillage
ret,img = cv2.threshold(gray_img,200,255,cv2.THRESH_BINARY)

"""
# === DILATE & ERODE === #
# Dilation
img = cv2.dilate(img, kernel, iterations=1)
# Erosion
img = cv2.erode(img, kernel, iterations=1)
"""

# === GETTING CONTOURS === #
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# areas = {}
# for i, contour in enumerate(contours):
#     area = cv2.contourArea(contour)
#     areas[i] = area
# # Filtering contours based on area
# num_coins = np.count_nonzero(areas) - 1

nombre_objets = len(contours)
print(f"Nombre d'objets détectés : {nombre_objets}")

cv2.drawContours(img, contours, -1, (0,255,0), 2)

# # Drawing bounds 
# for i in range(1, num_coins + 1):
#     cv2.drawContours(img, [contours[i]], -1, (0, 255, 0), 3)


# === DISPLAY IMAGE === #

# Show the image
cv2.imshow('photo_dice',img)

# Doesn't close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

