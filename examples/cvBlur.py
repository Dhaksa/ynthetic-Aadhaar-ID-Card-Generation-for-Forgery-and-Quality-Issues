import cv2
image = cv2.imread('check.png')
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
cv2.imwrite('blurred_imagecheckCV.jpg', blurred_image)
