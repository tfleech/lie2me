import torch
import torch.nn as nn
import torch.optim as optim
import cv2

from model import *

input_images = []

def get_data():
	vc = cv2.VideoCapture('./test_scripts/test_clip.mov')
	if vc.isOpened():
		rval, frame = vc.read()
	else:
		rval = False

	while rval:
		rval, frame = vc.read()
		input_images.append(frame)

def run():

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = FullNetwork()
	model = model.to(device)

	inp = np.array([input_images[0]], dtype='float32')
	inp = np.swapaxes(inp, 1,3)
	inp = np.swapaxes(inp, 2,3)

	inp = torch.from_numpy(inp)

	return model(inp)



get_data()
#print(len(input_images))
#print(input_images[0].shape)
run()