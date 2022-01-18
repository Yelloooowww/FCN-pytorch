#!/usr/bin/env python3
import numpy as np
import cv2
import roslib
import rospy
import struct
import math
import time
import os
import rospkg
import math
import time
import sys
import PIL
import pandas as pd
import scipy.misc
import random
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2, PointField
from geometry_msgs.msg import PoseArray, PoseStamped, Point
from nav_msgs.msg import Path
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header, Float32MultiArray
import message_filters
from datetime import datetime

from torchvision import transforms, utils, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from torchvision.models.vgg import VGG
from sklearn.metrics import confusion_matrix
from skimage import io, color


class FCN16s(nn.Module):

	def __init__(self, pretrained_net, n_class):
		super(FCN16s, self).__init__()
		self.n_class = n_class
		self.pretrained_net = pretrained_net
		self.relu    = nn.ReLU(inplace = True)
		self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn1     = nn.BatchNorm2d(512)
		self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn2     = nn.BatchNorm2d(256)
		self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn3     = nn.BatchNorm2d(128)
		self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn4     = nn.BatchNorm2d(64)
		self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn5     = nn.BatchNorm2d(32)
		self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

	def forward(self, x):
		output = self.pretrained_net(x)

		# After the feature extraction layer of vgg, you can get the feature map.
		# The size of the feature map after 5 max_pools are respectively
		x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
		x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
		x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
		x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
		x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

#===========FCN16s model ==========================
		score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
		score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
		score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
		score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
		score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
		score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
		score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
		return score  # size=(N, n_class, x.H/1, x.W/1)

class VGGNet(VGG):
	def __init__(self, cfg, pretrained=False, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
		super(VGGNet, self).__init__(self.make_layers(cfg[model]))
		ranges = {
			'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
			'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
			'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
			'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
		}
		self.ranges = ranges[model]

		if pretrained:
			exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

		if not requires_grad:
			for param in super().parameters():
				param.requires_grad = False

		if remove_fc:  # delete redundant fully-connected layer params, can save memory
			del self.classifier

		if show_params:
			for name, param in self.named_parameters():
				print(name, param.size())

	def forward(self, x):
		output = {}

		# get the output of each maxpooling layer (5 maxpool in VGG net)
		for idx in range(len(self.ranges)):
			for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
				x = self.features[layer](x)
			output["x%d" % (idx+1)] = x
		return output

	def make_layers(self, cfg, batch_norm=False):
		layers = []
		in_channels = 3
		for v in cfg:
			if v == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
				if batch_norm:
					layers += [conv2d,
							   nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
				else:
					layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = v
		return nn.Sequential(*layers)


class FCN_door_detector():
	def __init__(self):
		self.model = rospy.get_param("~model", "FCNs_door_batch8_epoch51_RMSprop_lr0.0001.pkl")
		my_dir = os.path.abspath(os.path.dirname(__file__))
		model_path = os.path.join(my_dir, "../weights/" + self.model)

		self.bridge = CvBridge()
		self.cfg = {
			'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
			'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
			'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
			'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
		}
		self.means = np.array([103.939, 116.779, 123.68]) / \
			255.  # mean of three channels in the order of BGR
		self.h, self.w = 320, 640
		# self.resize_count = 0
		self.n_class = 2

		self.vgg_model = VGGNet(self.cfg, requires_grad=True, remove_fc=True)
		self.fcn_model = FCN16s(
			pretrained_net=self.vgg_model, n_class=self.n_class)

		use_gpu = torch.cuda.is_available()
		num_gpu = list(range(torch.cuda.device_count()))
		rospy.loginfo("Cuda available: %s", use_gpu)

		if use_gpu:
			ts = time.time()
			self.vgg_model = self.vgg_model.cuda()
			self.fcn_model = self.fcn_model.cuda()
			self.fcn_model = nn.DataParallel(
				self.fcn_model, device_ids=num_gpu)
			print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

		self.fcn_model.load_state_dict(torch.load(model_path))

		self.mask1 = np.zeros((self.h, self.w))
		self.brand = ['', 'door']
		rospy.loginfo("FCN_door_detector ready!")

		self.image_sub = rospy.Subscriber("camera/image_raw", Image, self.img_cb)
		self.rgb_pub = rospy.Publisher("FCN_predict_img", Image, queue_size=1)
		self.mask_pub = rospy.Publisher("FCN_mask", Image, queue_size=1)
		self.arr_pub = rospy.Publisher("door_detect", Float32MultiArray, queue_size=1)
		rospy.loginfo("Start Predicting image")
		self.rgb_data = None


	def img_cb(self, rgb_data):
		self.rgb_data = rgb_data
		if self.rgb_data is not None:
			cv_image = self.bridge.imgmsg_to_cv2(self.rgb_data, "bgr8")
			generate_img, predict_img, cX, cY, obj_list = self.predict(cv_image)
			generate_img[generate_img == 1] = 255 # door
			self.rgb_pub.publish(self.bridge.cv2_to_imgmsg(predict_img, "bgr8"))
			self.mask_pub.publish(self.bridge.cv2_to_imgmsg(generate_img, "8UC1"))
			self.rgb_data = None


	def predict(self, img):
		img = cv2.resize(img, (640, 320), interpolation=cv2.INTER_CUBIC)
		rgb_predict = img
		img = img[:, :, ::-1]  # switch to BGR

		img = np.transpose(img, (2, 0, 1)) / 255.
		img[0] -= self.means[0]
		img[1] -= self.means[1]
		img[2] -= self.means[2]

		# convert to tensor
		img = img[np.newaxis, :]
		img = torch.from_numpy(img.copy()).float()

		output = self.fcn_model(img)
		output = output.data.cpu().numpy()

		N, _, h, w = output.shape
		mask = output.transpose(0, 2, 3, 1)
		mask = mask.reshape(-1, self.n_class).argmax(axis=1)
		mask = mask.reshape(N, h, w)[0]
		mask = np.asarray(mask, np.uint8)
		# =======================Filter=================================
		cnts, _ = cv2.findContours(
			mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]

		obj_list = []
		cX = 0
		cY = 0

		result_img = np.zeros(rgb_predict.shape)
		for c in cnts:
			M = cv2.moments(c)
			if M["m00"] == 0:
				break
			# cX = int(M["m10"] / M["m00"])
			# cY = int(M["m01"] / M["m00"])
			area = cv2.contourArea(c)

			if area > True:#4700:# ====Modify====
				print(area)
				cv2.drawContours(rgb_predict,[c],-1,(0, 255, 255), -1)
				cv2.drawContours(result_img,[c],-1,(255, 255, 255), -1)
				# cv2.circle(rgb_predict, (cX, cY), 7, (255, 0, 0), -1)
				# cv2.putText(rgb_predict, "door", (
				# 	cX-50, cY-40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (252, 197, 5), 3)

		result = Float32MultiArray()
		result.data = result_img[160,:,0]
		self.arr_pub.publish(result)
		return mask, rgb_predict, cX, cY, obj_list


	def mask_color_img(self, img, mask, color=[0, 255, 255], alpha=0.3):
		'''
		img: cv2 image
		mask: bool or np.where
		color: BGR triplet [_, _, _]. Default: [0, 255, 255] is yellow.
		alpha: float [0, 1].

		Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
		'''
		out = img.copy()
		img_layer = img.copy()
		img_layer[mask] = color
		out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
		return(out)

	def onShutdown(self):
		rospy.loginfo("Shutdown.")


if __name__ == '__main__':
	rospy.init_node('FCN_door_detector')
	foo = FCN_door_detector()
	rospy.on_shutdown(foo.onShutdown)
	rospy.spin()
