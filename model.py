import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ImageBlock(nn.Module):

	def __init__(self, in_channels, out_channels, stride=1):
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

	def forward(self, x):

		output = self.Conv1(x)
		output = F.relu(self.BN1(output))

		output = self.Conv2(output)
		output = self.BN2(output)

		output = F.relu(output)

		output = self.avgpool(output)
		output = output.view(x.size(0), -1)
		return output


class FullNetwork(nn.Module):
	def __init__(self):
		super(FullNetwork, self).__init__()

		self.image_layer = nn.Sequential(ImageBlock(3, 32))

	def forward(self,x):
		output = self.image_layer(x)
		return output