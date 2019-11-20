#!/usr/bin/env python
from __future__ import print_function
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
        print("Initializing")

        # Create subscriber nodes for master class
        rospy.Subscriber("/R1/pi_camera/image_raw", Image,
                         self.camera_callback, queue_size=1)
        rospy.Subscriber("/navigation", Int8,
                         self.navigation_callback, queue_size=1)
        rospy.Subscriber("/imProcessing", Bool,
                         self.imProcessing_callback, queue_size=1)
        rospy.Subscriber("/pedestrian", Bool,
                         self.imProcessing_callback, queue_size=1)

        # Set member variables
        self.vel_pub = rospy.Publisher('/R1/skid_vel', Twist, queue_size=1)
        self.nav_pub = rospy.Publisher('/navigation', Int8, queue_size=1)
        self.improcess_pub = rospy.Publisher('/imProcessing', Bool, queue_size=1)
        self.pedestrian_pub = rospy.Publisher('/pedestrian', Bool, queue_size=1)

        self.bridge = CvBridge()

        self.lines = None
        self.cv_image = None
        self.boundedImage = None

        # conditions
        self.seeRed = False
        self.seePedestrian = False
        self.seeCar = False

        # action
        self.Navigation = 0

        # globals
        self.rightEdge = True
        self.hysteresisSize = 30

    def navigation_callback(self, selectNav):
        if(self.lines is None):
            return
        edge_1 = self.lines[-60:-40, :].copy()
        edge_2 = self.lines[-150:-130, :].copy()
        vel_cmd = Twist()
        if selectNav != 0:
            setpoint = 625
            if selectNav == 1:
                setpoint = 950
                edge_1[:, :500] = 0
                edge_2[:, :500] = 0
            elif selectNav == 2:
                setpoint = 250
                edge_1[:, 700:] = 0
                edge_2[:, 700:] = 0
            cX1, cY1 = COM(edge_1)
            cX2, cY2 = COM(edge_2)

            if cX1 == 0 or cX2 == 0:
                center_detour = setpoint - cX1 - cX2
            else:
                center_detour = setpoint - (cX1 + cX2) / 2

            if center_detour > self.hysteresisSize:  # LEFT
                vel_cmd.linear.x = 0.0
                vel_cmd.angular.z = 0.5
            elif center_detour < -1 * self.hysteresisSize:  # RIGHT
                vel_cmd.linear.x = 0.0
                vel_cmd.angular.z = -0.5
            else:
                vel_cmd.linear.x = 0.2
                vel_cmd.angular.z = 0
        else:
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.0

        self.vel_pub.publish(vel_cmd)

        cv2.imshow("camera", self.lines)
        cv2.waitKey(5)

    def imProcessing_callback(self, process):
        self.boundedImage = self.cv_image
        self.lines = find_lines(self.cv_image)
        if(np.sum(find_Red(self.cv_image[-150:-1, 300:500].copy())) == 0):
            self.seeRed = False
        else:
            self.seeRed = True
        self.boundedImage, self.seeCar = filter_cars(self.boundedImage)

    def pedestrian_callback(self, ifRed):
		print("ye")

		if(ifRed):
			self.Navigation = 0
		else:
			self.Navigation = 1

    def camera_callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.pedestrian_pub.publish(self.seeRed)
        self.improcess_pub.publish(True)

        self.nav_pub.publish(self.Navigation)


def main():
    rospy.init_node('Master', anonymous=True)

    master = Master()

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
