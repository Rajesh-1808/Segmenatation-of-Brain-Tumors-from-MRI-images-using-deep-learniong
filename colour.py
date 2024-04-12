import cv2
import numpy as np

cap = cv.VideoCapture(0)

while(1):
	_, frame = cap.read()
	# It converts the BGR color space of image to HSV color space
	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
	
	# Threshold of blue in HSV space
	lower_blue = np.array([60, 35, 140])
	upper_blue = np.array([180, 255, 255])

	# preparing the mask to overlay
	mask = cv.inRange(hsv, lower_blue, upper_blue)
	
	# The black region in the mask has the value of 0,
	# so when multiplied with original image removes all non-blue regions
	result = cv.bitwise_and(frame, frame, mask = mask)

	cv.imshow('frame', frame)
	cv.imshow('mask', mask)
	cv.imshow('result', result)
	
	cv.waitKey(0)

cv.destroyAllWindows()
cap.release()
