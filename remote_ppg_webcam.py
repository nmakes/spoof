import cv2
from time import clock, time, perf_counter
import numpy as np
import matplotlib.pyplot as plt

measure = time


roiX = [100, 300, 500]
roiY = [200, 200, 200]

numRoi = len(roiX)

roiW = [20 for _ in range(numRoi)]
roiH = [20 for _ in range(numRoi)]

hrlist = [[] for i in range(numRoi)]
l = [0 for _ in range(numRoi)]

def show_webcam(mirror=False):
	
	faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	cam = cv2.VideoCapture(0)

	global hrlist

	numFrames = 60
	i = 0
	f = 0
	T = 0

	start = 0
	end = 0

	global l

	graphFFT = plt.figure('ffts')

	region = {0:'left', 1:'face', 2:'right'}

	while True:

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

		ret_val, img = cam.read()
		scale = 1
		w = int (img.shape[1] * scale)
		h = int (img.shape[0] * scale)
		dim = (w,h)
		img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=10,
			minSize=(30, 30),
			flags=cv2.CASCADE_SCALE_IMAGE
		)

		# Draw a rectangle around the faces
		for (x, y, w, h) in faces:
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


		if mirror: 
			img = cv2.flip(img, 1)

		blue = img.copy()[:,:,0]
		green = img.copy()[:,:,1]
		red = img.copy()[:,:,2]

		red = (blue+green+red)/3

		two = img.copy()
		two[:,:,0] *= 0
		two[:,:,2] *= 0

		for i in range(numRoi):
			cv2.rectangle(two, (roiX[i], roiY[i]), (roiX[i] + roiW[i], roiY[i] + roiH[i]), (0, 255, 0), 2)

		cv2.imshow('my webcam', img)
		cv2.imshow('red', two)

		for i in range(numRoi):
			
			if l[i] < numFrames:
				
				roi = red[roiY[i]:roiY[i]+roiH[i], roiX[i]:roiX[i]+roiW[i]]
				hrlist[i].append( np.mean(roi) )
				l[i]+=1

			else:

				graphFFT.clear()

				for j in range(numRoi):
					

					fftlist = np.fft.fft(hrlist[j])
					# maxpos = np.argmax(fftlist)
					freqlist = np.fft.fftfreq(numFrames, d=(1/fps))
					print(j, fps)
					# print("PPG:", freqlist[maxpos])

					# print(freqlist)

					ax = plt.subplot(numRoi * 100 + 10 + j+1)
					# ax.set_title('FFT of region ' + region[j])
					plt.plot(freqlist, fftlist.imag)

					l[j] = 0
					hrlist[j] = []
				
				plt.draw()
				plt.pause(0.001)

		if cv2.waitKey(1) == 27: 
			break  # esc to quit
	
	cv2.destroyAllWindows()


def main():
	show_webcam(mirror=False)


if __name__ == '__main__':
	main()
