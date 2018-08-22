import cv2
from time import clock, time, perf_counter
import numpy as np
import matplotlib.pyplot as plt

measure = time
numRoi = 1
hrlist = []
l = 0

def show_webcam(mirror=False):

	# Initialize Camera
	cam = cv2.VideoCapture(0)
	faces = []
	found = False # Flag to check whether face was found previously (if yes, then search in the local region)

	# Load cascades
	faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	graphFFT = plt.figure('ffts')
	global hrlist
	global l

	# Frame Rate Control
	numFrames = 32
	i = 0
	f = 0
	T = 0
	start = 0
	end = 0

	# Face Extraction Rate
	roiExLimit = 10
	roiExCount = 0

	# If face is not found for notFoundLimit turns, then reset the ROI
	notFoundLimit = 10
	notFoundCount = 0

	# Defining Face-coordinates and ROI coordinates
	X, Y, W, H = 0,0,16,16
	ROI_X, ROI_Y, ROI_W, ROI_H = 0,0,0,0

	while True:

		# Update face extraction counter
		roiExCount += 1

		# Calculate Frame Rate
		f += 1
		if(f==numFrames):
			f = 0
			end = measure()
			d = end - start
			if d!=0:
				fps = numFrames / d
				if (fps > 1):
					print('fps:', int(fps))
			start = measure()
			T = 0

		# Read Camera
		ret_val, img = cam.read()

		# Extract Face at the rate given by 1/roiExLimit
		if(roiExCount==roiExLimit):

			# Reset roiExCount
			roiExCount = 0

			# Extract the gray image in the ROI
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			# Find faces
			faces = faceCascade.detectMultiScale(
				gray,
				scaleFactor=1.1,
				minNeighbors=5,
				minSize=(30, 30),
				flags=cv2.CASCADE_SCALE_IMAGE
			)

		# Draw a rectangle around the faces & ROI
		for (x, y, w, h) in faces:

			# Draw rectangle around the detected face region
			cv2.rectangle(img, (x+ROI_X, y+ROI_Y), (x+ROI_X+w, y+ROI_Y+h), (0, 255, 0), 2)

			print("DET:", x,y,w,h)
			print()

			# Update the forehead position
			X = int(x + float(w)/2)
			Y = int(y + float(h)/5)
			cv2.rectangle(img, (X, Y), (X+W, Y+H), (0, 0, 255), 2)

			break # Extract only one face

		# if mirror:
		# 	img = cv2.flip(img, 1)

		# blue = img.copy()[:,:,0]
		green = img.copy()[:,:,1]
		# red = img.copy()[:,:,2]

		# red = (blue+green+red)/3

		greenIm = img.copy()
		greenIm[:,:,0] *= 0
		greenIm[:,:,2] *= 0

		# cv2.imshow('my webcam', img)
		cv2.imshow('green', greenIm)

		#
		if l < numFrames:

			roi = green[Y:Y+H, X:X+W]
			hrlist.append( np.mean(roi) )
			l+=1

		else:

			del hrlist[0]
			l-=1
			roi = green[Y:Y+H, X:X+W]
			hrlist.append( np.mean(roi) )
			l+=1

			if float('nan') in hrlist:
				hrlist.remove(float('nan'))
				l -= 1

			graphFFT.clear()

			plt.subplot(211)
			plt.plot(hrlist)

			# print(hrlist)
			# print(len(hrlist))
			fftlist = np.fft.fft(hrlist)
			freqlist = np.fft.fftfreq(numFrames, d=( (1/fps)/60 ) )

			plt.subplot(212)
			plt.plot(freqlist, fftlist.imag)

			# l[0] = 0
			# hrlist[0] = []

			plt.draw()
			plt.pause(0.001)

		if cv2.waitKey(1) == 27:
			break  # esc to quit

	cv2.destroyAllWindows()


def main():
	show_webcam(mirror=False)


if __name__ == '__main__':
	main()
