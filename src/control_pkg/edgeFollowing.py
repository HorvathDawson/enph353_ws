#!/usr/bin/env python
from __future__ import print_function

from helperClasses.image_processing import hugh_lines
from helperClasses.image_processing import find_lines
from helperClasses.image_processing import find_Cars
from helperClasses.image_processing import filter_cars
from helperClasses.image_processing import find_Red
from helperClasses.image_processing import COM
from geometry_msgs.msg import Twist
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import imutils
from collections import deque
import time

class edge_following:

  def __init__(self):
	self.bridge = CvBridge()
	self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
	self.vel_pub = rospy.Publisher('/R1/skid_vel', Twist, queue_size=1)
	self.rightEdge = True
	self.progCount = 0
	self.carFound = False
	self.picCount = 0
	self.q = deque([],maxlen=3)
	self.waitForPed = False
	self.x_cur,self.y_cur,self.w_cur,self.h_cur = 0,0,0,0
	self.pedFound = False
	self.frameCounter = 0
	self.crossingCount = 0

  def callback(self,data):
	try:
	  cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
	except CvBridgeError as e:
	  print(e)

	redRegion = cv_image[-150:-1,575:625].copy()

	lines = find_lines(cv_image)

	edge_1 = lines[-60:-40,:].copy()
	edge_2 = lines[-150:-130,:].copy()

	if(self.rightEdge):
	  setpoint = 950
	  edge_1[:,:500] = 0
	  edge_2[:,:500] = 0
	else:
	  setpoint = 250
	  edge_1[:,700:] = 0
	  edge_2[:,700:] = 0

	cX1, cY1 = COM(edge_1)
	cX2, cY2 = COM(edge_2)

	cv2.circle(lines, ((cX1 + cX2)/2, 680), 30, (100, 100, 100), -1)
	cv2.circle(lines, (setpoint, 680), 30, (100, 100, 100), -1)

	if(self.progCount%5==0):
		filteredCar, self.carFound = filter_cars(cv_image)
		cv2.imshow("car_cam",filteredCar)
		cv2.waitKey(5)

	if cX1 == 0 or cX2 == 0:
		center_detour = setpoint - cX1 - cX2
	else:
		center_detour = setpoint - (cX1 + cX2)/2

	vel_cmd = Twist()
	if center_detour > 50:  # LEFT
		vel_cmd.linear.x = 0.0
		vel_cmd.angular.z = 0.5
	elif center_detour < -50:  # RIGHT
		vel_cmd.linear.x = 0.0
		vel_cmd.angular.z = -0.5
	else:
		vel_cmd.linear.x = 0.2
		vel_cmd.angular.z = 0	

	print((np.sum(find_Red(redRegion))))
	if(((np.sum(find_Red(redRegion))>0) or self.waitForPed) and self.progCount%3==0):
		vel_cmd.linear.x = 0
		vel_cmd.angular.z = 0 
		self.vel_pub.publish(vel_cmd)

		print("ye")

		if(self.waitForPed == False):
			self.frameCounter = 0
			self.waitForPed = True

		self.q.append(cv_image)
 
		if(len(self.q)==self.q.maxlen):
			w = 0
			h = 0

			background = cv2.cvtColor(self.q[0],cv2.COLOR_BGR2GRAY)
			background = cv2.GaussianBlur(background,(21,21),0)

			liveFeed = cv2.cvtColor(self.q[-1],cv2.COLOR_BGR2GRAY)
			liveFeed = cv2.GaussianBlur(liveFeed,(21,21),0)

			frameDelta = cv2.absdiff(background,liveFeed)
			thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

			kernel = np.ones((4,3), dtype=np.uint8)
			#thresh = cv2.erode(thresh, kernel, iterations=1)
			thresh = cv2.dilate(thresh,kernel, iterations=12)

			im2, ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			ctrs = sorted(ctrs,key=lambda ctr: cv2.boundingRect(ctr)[0])

			if(len(ctrs)!=0):
				ctr = ctrs[-1]
				x, y, w, h = cv2.boundingRect(ctr)
			else:
				x, y, w, h = 0,0,0,0

			cv_image = cv_image.copy()


			color = (0,255,0) #green
			if((w*h > (1200*700)/150) and ((w*h)<(1200*700/5)) and (abs(w-600) < 500) and (abs(h-350) < 150):
				self.frameCounter += 1
				if(self.frameCounter>10):
					self.pedFound = True
				self.x_cur,self.y_cur,self.w_cur,self.h_cur = x, y, w, h
				color = (0,0,255) #red
			else:
				if(self.pedFound):
					self.crossingCount += 1
					self.pedFound = False
					self.waitForPed = False
					self.frameCounter = 0
					vel_cmd.linear.x = 0.5
					vel_cmd.angular.z = 0 
					self.vel_pub.publish(vel_cmd)
					time.sleep(1.5)
					print("exiting")


			cv2.rectangle(cv_image,(self.x_cur,int(0.97*self.y_cur)),(self.x_cur+self.w_cur,int((0.97)*(self.y_cur+self.h_cur))),color,2)

			cv2.imshow("ye",cv_image)




	self.vel_pub.publish(vel_cmd)

	# lines = hugh_lines(cv_image)
	# lines_edges = cv2.addWeighted(cv_image, 0.3, lines, 1, 0)
	# cv2.imshow("hughline transform", lines_edges)
	# cv2.waitKey(5)


def main(args):
  rospy.init_node('edge_following', anonymous=True)
  edgeFollower = edge_following()
  try:
	rospy.spin()
  except KeyboardInterrupt:
	print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
