'''
	Written By Naveen Venkat, 2018
	nav.naveenvenkat@gmail.com
	github.com/nmakes/spoof
	Final Year Undergraduate, BITS Pilani

	This work is licenced under a Creative Commons Attribution-NonCommercial-ShareAlike 
	(CC BY-NC-SA) licence. 
'''

import cv2
import numpy as np
from camera import StableFaceCapture

class DiscreteCellExperiment:

	'''
		This experiment discretizes the image into cells of dimensions rows x cols.
	'''

	def __init__(self, rows=5, cols=5, channels=3):

		# inherit parameters
		self.rows = int(rows)
		self.cols = int(cols)
		self.channels = int(channels)

		# set up camera & buffer
		self.cap = StableFaceCapture()
		r,c = self.cap.getCamShape()
		self.buffer = np.zeros(shape=(r,c,self.channels), dtype=np.uint8) # 3 color channels

		# set cell width and heights
		self.cellH = int(r/self.rows)
		self.cellW = int(c/self.cols)

	def run(self):

		while(True):

			# Capture the image
			img = self.cap.getCapture()

			# For each cell
			for i in range(self.rows):
				for j in range(self.cols):
					for c in range(self.channels):
						# Take the mean of the corresponding portion of the image
						self.buffer[i*self.cellH:(i+1)*self.cellH, j*self.cellW:(j+1)*self.cellW, c] = int(np.mean(img[i*self.cellH:(i+1)*self.cellH, j*self.cellW:(j+1)*self.cellW, c]))

			# AVERAGE IMAGE
			cv2.imshow('average image', self.buffer)

			# DIFFERENCE IMAGE
			difim = np.abs(img - self.buffer)
			cv2.imshow('diff image', difim)

			# REDIF IMAGE
			redifim = np.abs(img - difim)
			cv2.imshow('redif image', redifim)

			# SQUARED IMAGE
			img = img * 2
			img = np.array( (img / img.max(axis=None) ) * 255, dtype=np.uint8 )
			cv2.imshow('pow image', img)

			if cv2.waitKey(1) == 27:
				break  # esc to quit


if __name__=='__main__':

	r, c = 30, 40
	exp = DiscreteCellExperiment(rows=r, cols=c)
	exp.run()