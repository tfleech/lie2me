import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import glob

from model import *

truth_folder = './truths'
lie_folder = './lies'

truth_files = glob.glob(truth_folder+'/*.mp4')
lie_files = glob.glob(lie_folder+'/*.mp4')

total_truth_data = []
total_lie_data = []

def break_into_window(segs, window_size, label):
	res = []
	for i in range(len(segs)-window_size):
		data = [segs[i:i+window_size], label]
		res.append(data)
	return res

def get_data(file, label):
	vid_segs = []
	vc = cv2.VideoCapture(file)
	if vc.isOpened():
		rval, frame = vc.read()
	else:
		rval = False
	while rval:
		rval, frame = vc.read()
		if type(frame) != type(None):
			vid_segs.append(format_image(frame))

	return break_into_window(vid_segs, 5, label)

'''
def get_data():

	for vid in glob.glob(truth_folder+'/*.mp4'):	
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

		total_truth_data.extend(break_into_window(vid_segs, 5, torch.Tensor([0,1])))

	for vid in glob.glob(lie_folder+'/*.mp4'):
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

		total_lie_data.extend(break_into_window(vid_segs, 5, torch.Tensor([1,0])))
'''

def format_image(img):

	img = cv2.resize(img, (854, 480), interpolation=cv2.INTER_LINEAR)
	inp = np.array([img], dtype='float32')
	inp = np.swapaxes(inp, 1,3)
	inp = np.swapaxes(inp, 2,3)

	inp = torch.from_numpy(inp)
	return inp

def run():

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = FullNetwork()
	model = model.to(device)

	#get_data()
	print("Starting Training")

	num_epochs = 1

	loss_function = nn.CrossEntropyLoss().to(device)
	optimizer = optim.SGD(model.parameters(), lr = 0.001)

	True_sample = True
	sample_count = 0

	for epoch in range(num_epochs):
		total_loss = 0

		data_sample = None
		if True_sample:
			data_sample = get_data(truth_files[sample_count], torch.Tensor([0,1]))
			True_sample = False
		else:
			data_sample = get_data(lie_files[sample_count], torch.Tensor([1,0]))
			True_sample = True
			sample_count += 1

		for i in range(len(data_sample)):
			print(i)
			data, label = data_sample[i]

			#print(torch.sum(data[0]))

			data = torch.stack(data)
			#print(data.size())
			data.to(device)
			label.to(device)

			optimizer.zero_grad()

			output = model(data)
			#print(output)

			label_vec = torch.LongTensor([int(label[1]==1)])
			#print(label_vec)
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
#get_data()
#print len(total_truth_data)
#print len(total_lie_data)







