import cv2
import json
import numpy as np
# import base64
# import logging
# import random
img_path = "D:/dummy pic.jpeg"
i=cv2.imread(img_path)
i=cv2.resize(i,(300,300))
g = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
n = cv2.bitwise_not(i)
def adjust_brightness(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    final_hsv = cv2.merge((h, s, v))
    brightened_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return brightened_image
def adjust_saturation(image, value):
    # Convert the image to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Convert the S channel to a float32 type for accurate multiplication
    s = s.astype(np.float32)

    # Multiply the S channel by the value (increase or decrease saturation)
    s = s * (1 + value / 100.0)

    # Clip the values to stay within 0-255 range and convert back to uint8
    s = np.clip(s, 0, 255).astype(np.uint8)

    # Combine the modified channels back
    final_hsv = cv2.merge((h, s, v))

    # Convert back to BGR color space
    saturated_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return saturated_image
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    
    # Calculate the unsharp mask
    sharpened = float(amount + 1) * image - float(amount) * blurred
    
    # Clip values to stay within valid range
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # Apply thresholding to control the level of sharpening
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    return sharpened
i1=adjust_brightness(i,50)
i2=adjust_saturation(i,80)
i3=g
i4=unsharp_mask(i)
cv2.imwrite("./editedByBrightness.png",i1)
cv2.imwrite("./original.png",i)
cv2.imwrite("./editedBySaturation.png",i2)
cv2.imwrite("./greyscale.png",i3)
cv2.imwrite("./distinct.png",i4)
cv2.imwrite("./negetive.png",n)
laplacian = cv2.Laplacian(g, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))
cv2.imwrite("./edgeByLaplacian.png",laplacian)
blurred_image = cv2.GaussianBlur(g, (5, 5), 1.4)
canny = cv2.Canny(blurred_image, threshold1=50, threshold2=150)
cv2.imwrite("./edgeByCanny.png",canny)
g4 = cv2.cvtColor(i4, cv2.COLOR_BGR2GRAY)
cv2.imwrite("./greyscaleDistinct.png",g4)
pic = i
text = 'Dogs'
org = (80, 50)  # Bottom-left corner of the text string in the image
font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
font_scale = 1  # Font scale
color = (101, 0, 255)  # Text color in BGR (blue, green, red)
thickness = 2  # Thickness of the lines used to draw the text

# Add the text to the image
cv2.putText(pic, text, org, font, font_scale, color, thickness)
cv2.imwrite("./imageWithText.png",pic)

hsv_image = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
hue_adjustment = 100  # Value to adjust the hue
hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_adjustment) % 180
ih = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
cv2.imwrite("./imageHue.png", ih)
#i1=cv2.resize(i1,(128,128))
#i2=adjust_brightness(i,-50)
#i2=cv2.resize(i2,(128,128))
#cv2.imshow("original",i)
#cv2.imshow("edited",i1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(g, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(i, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the output image with detected faces
cv2.imwrite('./Detected Faces.png', i)
