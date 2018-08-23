'''
	Written By Naveen Venkat
	nav.naveenvenkat@gmail.com
	github.com/nmakes/spoof
	Final Year Undergraduate, BITS Pilani

	This work is licenced under a Creative Commons Attribution-NonCommercial-ShareAlike 
	(CC BY-NC-SA) licence. 
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils import face_utils
import dlib

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

	def __init__(self, threshold=0.025, noDetectionLimit=5, foreheadSize=(15, 10), foreheadScale=0.07, cvArgs={'scaleFactor':1.1, 'minNeighbors':5, 'minSize':(30, 30),'flags':cv2.CASCADE_SCALE_IMAGE}):

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
		self.foreheadSize = foreheadSize
		self.foreheadScale = foreheadScale
		self.cvArgs = cvArgs

		# Initialize camera and cascade classifier
		self.cam = cv2.VideoCapture(0)
		self.detector=dlib.get_frontal_face_detector()
		self.camWidth = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
		self.camHeight = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
		self.camDiag = np.sqrt(self.camWidth**2 + self.camHeight**2)
		self.faceCascade = cv2.CascadeClassifier('etc/haarcascade_frontalface_alt.xml')

		# Region of Interest
		self.ROI = None # np.array([0,0,0,0]) # (x,y,w,h)

		# Face capture
		self.F = None # np.array([0,0,0,0]) # (x,y,w,h)

		# Counters for stable face detection
		self.noDetectionLimit = noDetectionLimit
		self.noDetectionCounter = 0


	def getCamDims(self):
		return (self.camWidth, self.camHeight)


	def withinThreshold(self, loc):

		dF = np.abs(self.F - np.array(loc)) / self.camDiag
		if np.all(dF <= self.threshold):
			return True
		else:
			return False


	def getCapture(self):

		ret_val, img = self.cam.read()
		return img


	def getFace(self, img=None):

		if img is None:
			img = self.getCapture()

		if self.ROI is None: # First detection attempt
			self.ROI = (0, 0, self.camWidth, self.camHeight) # (x,y,w,h)

		# Get the image inside the ROI
		roiImg = img[	int(self.ROI[1]) : int(self.ROI[1]) + int(self.ROI[3]), 
						int(self.ROI[0]) : int(self.ROI[0]) + int(self.ROI[2])]

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

		# If no faces are found, we need to increase the noDetectionCounter, but return the previous
		# detected face region so as to avoid spikes in the estimation
		if len(faces) == 0:
			
			self.noDetectionCounter += 1
			
			if(self.noDetectionCounter==self.noDetectionLimit):
				self.noDetectionCounter = 0
				self.ROI = (0, 0, self.camWidth, self.camHeight)

			return self.F

		# Otherwise, reset the noDetectionCounter & continue the execution
		else:
			self.noDetectionCounter = 0

		# For the captured face(s), get the position of the face
		# Note: x, y are with respect to the image fed to the classifier. Thus,
		# these will be relative to the region of interest, and hence we add
		# ROI values to the x & y values.
		for (x,y,w,h) in faces:

			# If it is the first detection of a face, simply set the variables
			if(self.F is None): 
				self.F = np.array([self.ROI[0] + x, self.ROI[1] + y, w, h])
				# self.ROI = np.array([self.F[0] - 40, self.F[1] - 40, self.F[2]+80, self.F[3] + 80])
				return self.F

			# Otherwise, check the threshold
			else:

				# If the new region is within the threshold, return the old value
				if( self.withinThreshold((x,y,w,h)) ):
					return self.F

				# Otherwise, return the new region, while setting it to be the face region
				else:
					self.F = np.array([self.ROI[0] + x, self.ROI[1] + y, w, h])
					# self.ROI = np.array([self.F[0] - 40, self.F[1] - 40, self.F[2]+80, self.F[3] + 80])
					return self.F


	def getForehead(self, faceLoc, mode='absolute'):
		
		'''
			mode:
				- "absolute": indicates we want to extract a box of dimensions given by foreheadSize
				- "relative": indicates that the dimensions of the box will be with respect to the size
				of the detected face.
		'''

		(x,y,w,h) = faceLoc

		if (mode=='absolute'):
			X = int(x + float(w)/2 - (self.foreheadSize[0]/2))
			Y = int(y + float(h)/5)
			return (X,Y,self.foreheadSize[0], self.foreheadSize[1])

		elif (mode=='relative'):
			f = self.foreheadScale
			X = int(x + float(w)/2 - (self.foreheadSize[0]/2) - int( (w*f)/2))
			Y = int(y + float(h)/5)
			return (X, Y, int(w*f), int(h*f))

		else:
			raise ValueError('"mode" argument can only be either "absolute" or "relative"')


class StableKeypointExtractor:

	def __init__(self, dims, landmarks=5, threshold=0.02, useDlibDetector=True):
		
		self.useDlibDetector = useDlibDetector
		self.detector = dlib.get_frontal_face_detector()		
		self.predictor = dlib.shape_predictor('etc/shape_predictor_' + str(landmarks) + '_face_landmarks.dat')
		self.threshold = threshold
		self.camWidth = dims[0]
		self.camHeight = dims[1]
		self.camDiag = np.sqrt(dims[0]**2 + dims[1]**2)
		self.shape = None
		self.rect = None


	def withinThreshold(self, shape):
		ts = np.abs( (self.shape - shape) / self.camDiag )
		if (np.all(ts <= self.threshold)):
			return True
		else:
			return False


	def detectKeypoints(self, img, loc):
		
		# Set camera dimensions

		# Convert data for dlib
		if (self.useDlibDetector):
			rects = self.detector(img, 0)
			if(len(rects)>0):
				self.rect = rects[0]
		else:
			(x,y,w,h) = loc
			(x,y,w,h) = (x,y,w+20,h+20)
			self.rect = dlib.rectangle(x, y, x+w, y+h)

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		if self.rect is not None:

			# If it is the first extraction, set the shape and return it
			if self.shape is None:
				self.shape = self.predictor(gray, self.rect)
				self.shape = face_utils.shape_to_np(self.shape)

			# Otherwise, check for threshold bounds and return the appropriate shape
			else:
				tempshape = self.predictor(gray, self.rect)
				tempshape = face_utils.shape_to_np(tempshape)

				if self.withinThreshold(tempshape):
					return self.shape
				else:
					self.shape = tempshape

			return self.shape

		else:

			return []


# DEMO
if __name__=='__main__':

	cap = StableFaceCapture(threshold=0.025)
	ke = StableKeypointExtractor(dims=cap.getCamDims(), landmarks=68, threshold=0.02)

	while(True):

		img = cap.getCapture()
		loc = cap.getFace(img)

		if loc is not None:
			
			# Grab the forehead and keypoints
			(x, y, w, h) = loc
			(fx, fy, fw, fh) = cap.getForehead(loc, mode='relative')
			shape = ke.detectKeypoints(img, loc)

			# Draw the keypoints
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
			cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)
			for (X, Y) in shape:
				cv2.circle(img, (X, Y), 1, (0, 0, 255), -1)
			
			# Show the image
			cv2.imshow('camera', img)

		else:
			cv2.imshow('camera', img)
			pass

		if cv2.waitKey(1) == 27: 
			break  # esc to quit