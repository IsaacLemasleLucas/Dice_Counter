import cv2
import numpy as np

kernel = np.ones((5, 5), np.uint8)

# === GET THE IMAGE === #
# Read the image and stock it
img = cv2.imread("Dice.jpg", cv2.IMREAD_COLOR)
img2 = cv2.imread("Dice.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(img, (1200,800))
img2 = cv2.resize(img2,(1200,800))
# === THRESHOLD THE IMAGE === #
# Blurring the image
blurr_img = cv2.GaussianBlur(img,(7,7),3)
# Mise en noir et blanc 
gray_img = cv2.cvtColor(blurr_img, cv2.COLOR_BGR2GRAY)
# Seuillage
ret,thresh_img = cv2.threshold(gray_img,200,255,cv2.THRESH_BINARY)
# === DILATE & ERODE === #
# Dilation
dilate_img = cv2.dilate(thresh_img, kernel, iterations=1)
# Erosion
erode_img = cv2.erode(dilate_img, kernel, iterations=1)


# === GETTING CONTOURS === #
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours2, hierarchy = cv2.findContours(erode_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

nombre_objets = len(contours)
nombre_objets2 = len(contours2)
print(f"Nombre d'objets détectés : {nombre_objets}")
print(f"Nombre d'objets détectés 2 : {nombre_objets2}")

cv2.drawContours(img, contours, -1, (0,255,0), 3)
cv2.drawContours(img2, contours2, -1, (0,255,0), 3)


# === GETTING AIRES === #
seuil_aire_min = 360

for i, contour in enumerate(contours):
    aire = cv2.contourArea(contour)
    if aire > seuil_aire_min:
        # print(f"Contours #{i} - Aire : {aire} pixels")
        cv2.drawContours( img, [contour], -1, (255,0,0),3)

for i, contour in enumerate(contours2):
    aire = cv2.contourArea(contour)
    if aire > seuil_aire_min:
        # print(f"Contours #{i} - Aire : {aire} pixels")
        cv2.drawContours( img2, [contour], -1, (255,0,0),3)

# === DISPLAY IMAGE === #

# Show the image
cv2.imshow('photo_dice',img)
cv2.imshow('photo_dice2',img2)


# Doesn't close the window immediately
cv2.waitKey(0)
cv2.destroyAllWindows()

