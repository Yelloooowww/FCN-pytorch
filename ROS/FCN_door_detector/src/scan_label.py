#!/usr/bin/env python3
import numpy as np
import cv2
import roslib
import rospy
# import tf
import struct
import math
import time
import os
import rospkg
import math
import time
import sys
import PIL
import random
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseArray, PoseStamped, Point
from nav_msgs.msg import Path
# from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
import message_filters
from datetime import datetime
from std_msgs.msg import Header, Float32MultiArray
from scipy import interpolate

class Scan_FCN_Label():
	def __init__(self):
		self.scan_sub = rospy.Subscriber("/husky2/RL/scan", LaserScan, self.scan_cb)
		self.result_sub = rospy.Subscriber("door_detect", Float32MultiArray, self.result_cb)
		self.scan_pub = rospy.Publisher("RL/scan_label", LaserScan, queue_size=1)
		self.scan = None
		rospy.loginfo("LaserScan_FCN_Label init!")


	def result_cb(self, msg):
		if not self.scan==None:
			raw_Data = msg.data
			raw_Data_360 = []
			for i in range(360):
				raw_Data_360.append(raw_Data[-int(i*640/360)])
			intensities = [raw_Data_360[-i] for i in range(360)]
			# for i in range(241): intensities.append(raw_Data_360[i+60])
			scan_label = self.scan
			scan_label.intensities = raw_Data_360
			self.scan_pub.publish(scan_label)
		self.scan = None

	def scan_cb(self, msg):
		self.scan = msg


	def onShutdown(self):
		rospy.loginfo("Shutdown.")


if __name__ == '__main__':
	rospy.init_node('Scan_FCN_Label')
	foo = Scan_FCN_Label()
	rospy.on_shutdown(foo.onShutdown)
	rospy.spin()
