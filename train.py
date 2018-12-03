import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import glob

from model import *

truth_folder = './truths'
lie_folder = './lies'

total_truth_data = []
total_lie_data = []

def break_into_window(segs, window_size, label):
	res = []
	for i in range(len(segs)-window_size):
		data = [segs[i:i+window_size], label]
		res.append(data)
	return res

def get_data():

	for vid in glob.glob(truth_folder+'/*.mov'):
		vid_segs = []
		vc = cv2.VideoCapture(vid)
		if vc.isOpened():
			rval, frame = vc.read()
		else:
			rval = False
		while rval:
			rval, frame = vc.read()
			if type(frame) != type(None):
				vid_segs.append(format_image(frame))
		vc.release()

		total_truth_data.extend(break_into_window(vid_segs, 5, [0,1]))

	for vid in glob.glob(lie_folder+'/*.mov'):
		vid_segs = []
		vc = cv2.VideoCapture(vid)
		if vc.isOpened():
			rval, frame = vc.read()
		else:
			rval = False
		while rval:
			rval, frame = vc.read()
			if type(frame) != type(None):
				vid_segs.append(format_image(frame))
		vc.release()

		total_lie_data.extend(break_into_window(vid_segs, 5, [1,0]))

def format_image(img):
	inp = np.array([img], dtype='float32')
	inp = np.swapaxes(inp, 1,3)
	inp = np.swapaxes(inp, 2,3)

	inp = torch.from_numpy(inp)
	return inp

def run():

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = FullNetwork()
	#model = model.to(device)

	get_data()

	num_epochs = 1

	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr = 0.05)

	for epoch in range(num_epochs):
		total_loss = 0

		for i in range(len(total_truth_data)):
			print(i)
			data, label = total_truth_data[i]

			optimizer.zero_grad()
			output = model(data)

			label_vec = torch.LongTensor([label[1]==1])
			#print(label_vec.size())
			#print(output.size())
			loss = loss_function(output, label_vec)
			loss.backward()
			total_loss += loss.item()
			optimizer.step()
			print(total_loss)



#print(len(total_truth_data))
#print(len(total_truth_data[0]))
#print(len(total_truth_data[0][0]))
#print(total_truth_data[0][0][0].shape)
run()