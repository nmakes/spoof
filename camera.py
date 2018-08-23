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

	def __init__(self, threshold=0.05, noDetectionLimit=5, cvArgs={'scaleFactor':1.1, 'minNeighbors':5, 'minSize':(30, 30),'flags':cv2.CASCADE_SCALE_IMAGE}):

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
		self.camWidth = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
		self.camHeight = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
		self.camDiag = np.sqrt(self.camWidth**2 + self.camHeight**2)
		self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

		# Region of Interest
		self.ROI = None # np.array([0,0,0,0]) # (x,y,w,h)

		# Face capture
		self.F = None # np.array([0,0,0,0]) # (x,y,w,h)

		# Counters for stable face detection
		self.noDetectionLimit = noDetectionLimit
		self.noDetectionCounter = 0


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

		# If no faces are found, we need to increase the noDetectionCounter
		if len(faces) == 0:
			
			self.noDetectionCounter += 1
			
			if(self.noDetectionCounter==self.noDetectionLimit):
				self.noDetectionCounter = 0
				self.ROI = (0, 0, self.camWidth, self.camHeight)

			return None

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
				return self.F

			# Otherwise, check the threshold
			else:

				# If the new region is within the threshold, return the old value
				if( self.withinThreshold((x,y,w,h)) ):
					return self.F

				# Otherwise, return the new region, while setting it to be the face region
				else:
					self.F = np.array([x,y,w,h])
					return self.F


cap = StableFaceCapture()

while(True):

	img = cap.getCapture()
	loc = cap.getFace(img)

	if loc is not None:
		(x,y,w,h) = loc
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.imshow('camera', img)
	else:
		cv2.imshow('camera', img)
		pass

	if cv2.waitKey(1) == 27: 
		break  # esc to quit