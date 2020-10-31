import cv2 as cv
import imutils
import numpy as np
import pytesseract
import string

# image = cv.imread('C:/Users/Navin Subbu/Documents/BEEP/Project/models/research/object_detection/detected/September/CAP_22Sep_06-44PM.png',cv.IMREAD_COLOR)
image = cv.imread('C:/Users/Navin Subbu/Documents/BEEP/Code Snippets/og.png',cv.IMREAD_COLOR)

#image = cv.resize(image, (640,480),interpolation = cv.INTER_AREA)
cv.imshow('Original Image',image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #convert to grey scale
cv.imshow('Grayscale Image',gray)


gray = cv.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise
cv.imshow('Bilateral Filter',gray)


edge = cv.Canny(gray, 30, 200) #Perform Edge detection
cv.imshow('Canny Edge Detection',edge)


# Retaining only the contour with number plate
contours = cv.findContours(edge.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv.contourArea, reverse = True)[:10]
count = None

# loop over our contours
for c in contours:
 # approximate the contour
 peri = cv.arcLength(c, True)
 approx = cv.approxPolyDP(c, 0.018 * peri, True)
 
 # if our approximated contour has four points, then
 # we can assume that we have found our screen
 if len(approx) == 4:
  count = approx
  break

if count is None:
 detected = 0
 print("No contour detected")
else:
 detected = 1

if detected == 1:
 cv.drawContours(image, [count], -1, (0, 255, 0), 3)

# Masking the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv.drawContours(mask,[count],0,255,-1,)
new_image = cv.bitwise_and(image,image,mask=mask)
cv.imshow('Mask',new_image)

# Now crop
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]
cv.imshow('image',image)
cv.imshow('Cropped',Cropped)




#Read the number plate
text = pytesseract.image_to_string(Cropped, config='--psm 11')
print("Detected Number is:",text)
 

whitelist = string.digits + string.ascii_letters + ' '
new_string = ''
for char in text:
    if char in whitelist:
        new_string += char
    else:
        new_string += ''
        
print(new_string)

out = cv.putText(image, text, (420,450), cv.FONT_HERSHEY_SIMPLEX,1.25, (255,255,255), 1, cv.LINE_AA)
cv.imshow('Output', out)
cv.waitKey(0)
cv.destroyAllWindows()
