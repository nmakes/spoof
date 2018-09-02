'''
	Written By Naveen Venkat, 2018
	nav.naveenvenkat@gmail.com
	github.com/nmakes/spoof
	Final Year Undergraduate, BITS Pilani

	This work is licenced under a Creative Commons Attribution-NonCommercial-ShareAlike 
	(CC BY-NC-SA) licence. 
'''

import camera

# DEMO
if __name__=='__main__':

	cap = camera.StableFaceCapture(threshold=0.025)

	while(True):

		img = cap.getCapture()
		loc = cap.getFace(img)

		if loc is not None:
			
			# Grab the forehead and keypoints
			(x, y, w, h) = loc

			# Draw the keypoints
			camera.cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
			
		# Show the image
		camera.cv2.imshow('camera', img)

		if camera.cv2.waitKey(1) == 27:
			break  # esc to quit