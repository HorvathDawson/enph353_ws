#!/usr/bin/env python
from __future__ import print_function

from helperClasses.image_processing import hugh_lines
from helperClasses.image_processing import find_lines
from helperClasses.image_processing import find_Cars
from helperClasses.image_processing import filter_cars
from helperClasses.image_processing import COM
from geometry_msgs.msg import Twist
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class edge_following:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
    self.vel_pub = rospy.Publisher('/R1/skid_vel', Twist, queue_size=1)
    self.rightEdge = True
    self.progCount = 0
    self.carFound = False
  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    self.progCount += 1

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
    # to far right = dot on right of set point
    # put text and highlight the center
    # cv2.circle(lines, (cX1, 700), 30, (155, 155, 155), -1)
    # cv2.circle(lines, (cX2, 680), 30, (155, 155, 155), -1)
    cv2.circle(lines, ((cX1 + cX2)/2, 680), 30, (100, 100, 100), -1)
    cv2.circle(lines, (setpoint, 680), 30, (100, 100, 100), -1)
    #cv2.imshow("check_node", lines)
    #cv2.imshow("reg_cam",cv_image)

    if(self.progCount%5==0):
	self.progCount = 0
    	filteredCar, self.carFound = filter_cars(cv_image)
    	cv2.imshow("car_cam",filteredCar)
        cv2.waitKey(5)

    if cX1 == 0 or cX2 == 0:
        center_detour = setpoint - cX1 - cX2
    else:
        center_detour = setpoint - (cX1 + cX2)/2

    vel_cmd = Twist()
    if center_detour > 30:  # LEFT
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.5
    elif center_detour < -30:  # RIGHT
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = -0.5
    else:
        vel_cmd.linear.x = 0.2
        vel_cmd.angular.z = 0

    if(self.carFound):
        vel_cmd.linear.x = 0
        vel_cmd.angular.z = 0.5

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
