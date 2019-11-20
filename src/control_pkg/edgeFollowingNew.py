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
from std_msgs.msg import Bool, Int8, Int32, Float64

class Master():
	def __init__(self):
		#Subscribe to all nodes
		print("Initializing")
		#Set member variables
		self.vel_pub = rospy.Publisher('/R1/vel_cmd', Twist, queue_size=1)
		#self.navigationSelect = rospy.Publisher('NavigationSelect',Int8,queue_size=1)
		self.bridge = CvBridge()
		self.lines = None
		self.rightEdge = True
		

	def navigation_callback(self,selectNav):
		print("ye")
		if(self.lines is None):
			pass
		print("passed")
		edge_1 = self.lines[-60:-40,:].copy()
		edge_2 = self.lines[-150:-130,:].copy()

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

		self.vel_pub.publish(vel_cmd)	

		cv2.imshow("camera",lines)
		cv2.waitKey(5)

		return True

	def camera_callback(self,data):
		try:
		  	cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
		  	print(e)

		self.lines = find_lines(cv_image)

def main():
	rospy.init_node('control_master',anonymous=True)

	master = Master()

 	rospy.Subscriber("/R1/pi_camera/image_raw",Image,master.camera_callback,queue_size=1)
	rospy.Subscriber("Navigation",Bool,master.navigation_callback,queue_size=1)

	rospy.spin()


if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException: pass