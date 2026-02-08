import cv2

# Read the image.
img = cv2.imread('data/raw/HT7b.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply bilateral filter with d = 15, 
# sigmaColor = sigmaSpace = 75.
bilateral = cv2.bilateralFilter(img, 15, 75, 75)

# Save the output.
cv2.imwrite('data/HT7b-bilateral.jpg', bilateral)

thresh_gauss = cv2.adaptiveThreshold(
    bilateral, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    19, 5
)
cv2.imwrite('data/HT7b-thresh-gauss.jpg', thresh_gauss)