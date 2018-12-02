#test clip editing
import cv2

print('0')

vc = cv2.VideoCapture('test_clip.mov')

print('1')

if vc.isOpened():
	rval, frame = vc.read()
else:
	rval = False
	print('2')

while rval:
	rval, frame = vc.read()
	try:
		print frame.shape

	except:
		print "hi"
		
#result shape: (720, 1080, 3)

vc.release()