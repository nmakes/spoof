import cv2
import matplotlib.pyplot as plt
import numpy as np

class StableFaceCapture:

	'''
		Notes:

		1. OpenCV considers x & y to be horizontal (left-right) and vertical (top-bottom) 
		directions, while numpy considers x to be axis-0 (top-bottom) and y to be axis-1.

		2. We set a threshold to avoid unstable detection. We create a region of interest (ROI) which
		is slighly bigger than the detected face's bounding box. If face is detected again inside
		the ROI, we do not change it. If the face is not detected inside the ROI, we look for the face
		again in the whole image.

		3. As our work entails detecting spoof, we assume that only one face will be detected 
		in the frame.
	'''

	def __init__(	self, 
					threshold=0.05, 
					noDetectionLimit=5,
					cvArgs={'scaleFactor':1.1, 
							'minNeighbors':5, 
							'minSize':(30, 30),
							'flags':cv2.CASCADE_SCALE_IMAGE}
				):

		'''
			- threshold: 		this will ensure that if the captured face is within
								5% of the previously captured position, then the Fx,Fy,Fw,Fh
								will not be altered

			- noDetectionLimit:	if face is not detected for these many frames inside the ROI,
								we look for the face in the whole image

			- cvArgs={...}: 	arguments to opencv's detectMultiScale function
		'''

		# Inherit parameters
		self.threshold = threshold
		self.cvArgs = cvArgs

		# Initialize camera and cascade classifier
		self.cam = cv2.VideoCapture(0)
		self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

		# Region of Interest
		self.ROIx = None
		self.ROIy = None
		self.ROIw = None
		self.ROIh = None

		# Face capture
		self.Fx = None
		self.Fy = None
		self.Fw = None
		self.Fh = None

		# Counters for stable face detection
		self.noDetectionLimit = noDetectionLimit
		self.noDetectionCounter = 0


	def withinThreshold(x,y,w,h):

		dx = self.Fx - x
		dy = self.Fy - y
		dw = self.Fw - w
		dh = self.Fh - h


	def getCapture():

		ret_val, img = self.cam.read()
		return img


	def getFace(img):

		if self.ROIx is None: # First detection attempt

			self.ROIx = 0
			self.ROIy = 0
			self.ROIw = img.shape[1]
			self.ROIh = img.shape[0]

		# Get the image inside the ROI
		roiImg = img[self.ROIy : self.ROIy + self.ROIh, self.ROIx : self.ROIx + self.ROIh]

		# Get the gray image (for haar cascade classification)
		grayRoiImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Get the faces
		faces = self.faceCascade.detectMultiScale(
				grayRoiImg,
				scaleFactor=self.cvArgs['scaleFactor'],
				minNeighbors=self.cvArgs['minNeighbors'],
				minSize=self.cvArgs['minSize'],
				flags=self.cvArgs['flags']
			)

		# If no faces are found, we need to increase the noDetectionCounter
		if len(faces) == 0:
			
			noDetectionCounter += 1
			
			if(noDetectionCounter==noDetectionLimit):
				noDetectionCounter = 0
				self.ROIx = 0
				self.ROIy = 0
				self.ROIw = img.shape[1]
				self.ROIh = img.shape[0]

			return None

		# Otherwise, reset the noDetectionCounter & continue the execution
		else:
			noDetectionCounter = 0

		# For the captured face(s), get the position of the face
		for (x,y,w,h) in faces:

			# If it is the first detection of a face, simply set the variables
			if(self.Fx is None): 
				self.Fx = x
				self.Fy = y
				self.Fw = w
				self.Fh = h

			# Otherwise, check threshold and perform bounding box check
			else:


			break # Get only one face