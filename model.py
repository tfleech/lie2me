import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ImageBlock(nn.Module):

	def __init__(self, in_channels, out_channels, encoding_size, stride=1):
		super(ImageBlock, self).__init__()

		self.Conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.BN1 = nn.BatchNorm2d(out_channels)
		self.Conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.BN2 = nn.BatchNorm2d(out_channels)

		self.avgpool = nn.AvgPool2d(8)

		if stride != 1 or in_channels != out_channels:
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels),
			)

		self.fc = nn.Linear(6360*out_channels, encoding_size)

	def forward(self, x):

		output = self.Conv1(x)
		output = F.relu(self.BN1(output))

		output = self.Conv2(output)
		output = self.BN2(output)

		output = F.relu(output)

		output = self.avgpool(output)
		output = output.view(x.size(0), -1)
		output = self.fc(output)
		return output


class FullNetwork(nn.Module):
	def __init__(self):
		super(FullNetwork, self).__init__()

		self.out_channels = 32
		self.encoding_size = 1000
		self.window_size = 5
		self.class_size = 2

		self.image_layer = nn.Sequential(ImageBlock(3, self.out_channels, self.encoding_size))

		self.fc = nn.Linear(self.window_size*self.encoding_size, self.class_size)


	def forward(self,x):
		res = []
		for i in x:
			res.append(self.image_layer(i))

		output = torch.cat(res, 1)

		#output = nn.Sequential(self.fc, nn.dropout(0.2))
		output = self.fc(output)
		return output