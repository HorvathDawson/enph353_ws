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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import imutils
from collections import deque
import time
from std_msgs.msg import Bool, Int8, Int32, Float64, String
from licensePlateProcess.ParseCarImage import ParseCarImage


class Master():
    def __init__(self):
        print("Initializing")

        # Create subscriber nodes for master class
        rospy.Subscriber("/R1/pi_camera/image_raw", Image,
                         self.camera_callback, queue_size=1)
        rospy.Subscriber("/navigation", Bool,
                         self.navigation_callback, queue_size=1)
        rospy.Subscriber("/imProcessing", Bool,
                         self.imProcessing_callback, queue_size=1)
        rospy.Subscriber("/pedestrian", Bool,
                         self.pedestrian_callback, queue_size=1)
        self.findLicense_sub = rospy.Subscriber("/findLicense", Bool,
                         self.findLicense_callback, queue_size=1)

        # Set member variables
        self.license_pub = rospy.Publisher(
            '/license_plate', String, queue_size=1)
        self.vel_pub = rospy.Publisher('R1/cmd_vel', Twist, queue_size=10)
        self.nav_pub = rospy.Publisher('/navigation', Bool, queue_size=1)
        self.improcess_pub = rospy.Publisher(
            '/imProcessing', Bool, queue_size=1)
        self.pedestrian_pub = rospy.Publisher(
            '/pedestrian', Bool, queue_size=1)
        self.findLicense_pub = rospy.Publisher(
            '/findLicense', Bool, queue_size=1)

        self.bridge = CvBridge()

        self.lines = None
        self.cv_image = None
        self.boundedImage = None
        self.reductionMod = 3

        self.Running = True

        # conditions
        self.seeRed = False
        self.seePedestrian = False
        self.seeCar = False
        self.onCrosswalk = False
        self.blindToRed = False
        self.frameCounter = 0
        self.frameCountReached = False
        self.turn = False

        # action
        self.Navigation = False

        # globals
        self.rightEdge = False
        self.hysteresisSize = 40
        self.pedestrian_buffer = 0
        self.safeToGo = False
        self.frameCounter = 0

        self.x_cur,self.y_cur,self.w_cur,self.h_cur = 0,0,0,0
        self.speedReductionTimer = 0
        self.passedPedestrians = 0

        self.q = deque([], maxlen=3)
        self.x_cur, self.y_cur, self.w_cur, self.h_cur = 0, 0, 0, 0

    def navigation_callback(self, isRunning):
        self.speedReductionTimer += 1
        if(self.lines is None):
            return
        edge_1 = self.lines[-60:-40, :].copy()
        edge_2 = self.lines[-150:-130, :].copy()
        vel_cmd = Twist()

        if self.rightEdge == True:
            setpoint = 950
            edge_1[:, :500] = 0
            edge_2[:, :500] = 0
        else:
            setpoint = 125
            edge_1[:, 700:] = 0
            edge_2[:, 700:] = 0
        cX1, cY1 = COM(edge_1)
        cX2, cY2 = COM(edge_2)

        if cX1 == 0 or cX2 == 0:
            center_detour = setpoint - cX1 - cX2
        else:
            center_detour = setpoint - (cX1 + cX2) / 2

        if center_detour > self.hysteresisSize:  # LEFT
            if self.speedReductionTimer % self.reductionMod == 0:
                vel_cmd.linear.x = 0.0
                vel_cmd.angular.z = 0.5
            else:
                vel_cmd.linear.x = 0.0
                vel_cmd.angular.z = 0.0
        elif center_detour < -1 * self.hysteresisSize:  # RIGHT
            if self.speedReductionTimer % self.reductionMod == 0:
                vel_cmd.linear.x = 0.0
                vel_cmd.angular.z = -0.5
            else:
                vel_cmd.linear.x = 0.0
                vel_cmd.angular.z = 0.0
        else:
            vel_cmd.linear.x = 0.3
            vel_cmd.angular.z = 0

        if self.turn:
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
            self.vel_pub.publish(vel_cmd)
            time.sleep(0.1)
            self.turn = False

        if not self.Navigation:
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.0

        if self.Navigation and self.onCrosswalk:
            vel_cmd.linear.x = 0.5
            vel_cmd.angular.z = 0.0

        if not isRunning.data:
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.0

        self.vel_pub.publish(vel_cmd)

    def imProcessing_callback(self, isRunning):
        if not isRunning.data:
            return
        self.boundedImage = self.cv_image
        self.lines = find_lines(self.cv_image)
        if(np.sum(find_Red(self.cv_image[-50:-1, 300:500].copy())) > 200):
            self.seeRed = True
        else:
            self.seeRed = False
        self.boundedImage, self.seeCar = filter_cars(self.boundedImage)
        if self.seeCar:
            self.rightEdge = True
            self.blindToRed = False

    def findLicense_callback(self, isRunning):
        if isRunning.data:
            return
        arr = ParseCarImage()
        for i in range(len(arr)):
            print(arr[i])
            self.license_pub.publish(arr[i])
        self.findLicense_sub.unregister()
        print("processinglicense")

    def pedestrian_callback(self, isRunning):
        if not isRunning.data:
            return
        if self.onCrosswalk:
            # do actions to deal with it
            self.pedestrian_buffer += 1
            self.blindToRed = True
            if self.seePedestrian or self.pedestrian_buffer < 50:
                self.q.append(self.cv_image)
                if(len(self.q) == self.q.maxlen):
                    w = 0
                    h = 0

                    background = cv2.cvtColor(self.q[0], cv2.COLOR_BGR2GRAY)
                    background = cv2.GaussianBlur(background, (21, 21), 0)

                    liveFeed = cv2.cvtColor(self.q[-1], cv2.COLOR_BGR2GRAY)
                    liveFeed = cv2.GaussianBlur(liveFeed, (21, 21), 0)

                    frameDelta = cv2.absdiff(background, liveFeed)
                    thresh = cv2.threshold(
                        frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

                    kernel = np.ones((4, 3), dtype=np.uint8)
                    thresh = cv2.dilate(thresh, kernel, iterations=12)

                    cv2MajorVersion = cv2.__version__.split(".")[0]
                    if int(cv2MajorVersion) == 4:
                        ctrs, hier = cv2.findContours(
                            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    else:
                        im2, ctrs, hier = cv2.findContours(
                            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    ctrs = sorted(
                        ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

                    if(len(ctrs) != 0):
                        ctr = ctrs[-1]
                        x, y, w, h = cv2.boundingRect(ctr)
                    else:
                        x, y, w, h = 0, 0, 0, 0

                    self.boundedImage = self.boundedImage.copy()

                    color = (0, 255, 0)  # green
                    if((w * h > (1200 * 700) / 150) and ((w * h) < (1200 * 700 / 5)) and (abs(x + (w / 2) - 600) < 400) and (abs(y + (h / 2) - 350) < 250)):
                        self.seePedestrian = True
                        self.frameCounter += 1
                        if(self.frameCounter > 15):
                            self.frameCountReached = True
                        self.x_cur, self.y_cur, self.w_cur, self.h_cur = x, y, w, h
                        color = (0, 0, 255)  # red
                    else:
                        if(self.frameCountReached):
                            self.frameCounter = 0
                            self.frameCountReached = False
                            self.seePedestrian = False

                    cv2.rectangle(self.boundedImage, (self.x_cur, int(
                        0.97 * self.y_cur)), (self.x_cur + self.w_cur, int((0.97) * (self.y_cur + self.h_cur))), color, 2)

            elif self.onCrosswalk and not np.sum(self.lines[-250:-1, 550:650]):
                self.onCrosswalk = False
                self.passedPedestrians += 1
                self.Navigation = True
            else:
                self.Navigation = True

        elif self.seeRed and not self.blindToRed:
            if(self.passedPedestrians is 0 and not self.rightEdge):
                self.turn = True
            self.rightEdge = True
            self.Navigation = False
            self.onCrosswalk = True
            self.seePedestrian = True
        else:
            self.pedestrian_buffer = 0
            self.Navigation = True

        if self.boundedImage is not None:
            cv2.imshow("camera", self.boundedImage)
            cv2.waitKey(5)

    def camera_callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        if self.passedPedestrians > 1 and not self.blindToRed:
            self.Running = False
        self.pedestrian_pub.publish(self.Running)
        self.improcess_pub.publish(self.Running)
        self.nav_pub.publish(self.Running)
        self.findLicense_pub.publish(self.Running)

def main():
    rospy.init_node('Master', anonymous=True)
    master = Master()
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
