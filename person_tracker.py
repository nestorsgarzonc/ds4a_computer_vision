import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import pandas as pd
from datetime import datetime
import re
from sqlalchemy import create_engine, text
##for google collab
from google.colab.patches import cv2_imshow

flags.DEFINE_string('weights_path', './checkpoints/yolov4-416','path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video_path', '/content/drive/My Drive/Team_69_DS4A/Videos/Camara Floresta Recortado.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output_vid', './outputs/tracker.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score_th', 0.40, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('display_count', False, 'count objects being tracked on screen')
flags.DEFINE_string('output_csv_path', './outputs/video_data.csv','path to output csv file')
flags.DEFINE_string('startline', '193,183','start point of the line for the entrance of the store')
flags.DEFINE_string('endline', '650,183','end point of the line for the entrance of the store')

def main(_argv):
	# Definition of the parameters
	max_cosine_distance = 0.5
	nn_budget = None
	nms_max_overlap = 0.8
	
	# initialize deep sort
	model_filename = 'model_data/mars-small128.pb'
	encoder = gdet.create_box_encoder(model_filename, batch_size=1)
	# calculate cosine distance metric
	metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
	# initialize tracker
	tracker = Tracker(metric)
	
	# load configuration for object detector
	config = ConfigProto()
	config.gpu_options.allow_growth = True
	session = InteractiveSession(config=config)
	#STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
	input_size = FLAGS.size
	video_path = FLAGS.video_path
	startline = tuple(map(int,FLAGS.startline.split(',')))
	endline = tuple(map(int,FLAGS.endline.split(',')))
	output_csv_path = FLAGS.output_csv_path
	
	#Extract the date and time information from the video_path name
	time_start_vid, time_end_vid = re.findall('[0-9]{14}',video_path)
	
	saved_model_loaded = tf.saved_model.load(FLAGS.weights_path, tags=[tag_constants.SERVING])
	infer = saved_model_loaded.signatures['serving_default']
	
	# begin video capture
	try:
		vid = cv2.VideoCapture(int(video_path))
	except:
		vid = cv2.VideoCapture(video_path)
	
	out = None
	
	# get video ready to save locally if flag is set
	if FLAGS.output_vid:
		# by default VideoCapture returns float instead of int
		width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = int(vid.get(cv2.CAP_PROP_FPS))
		codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
		out = cv2.VideoWriter(FLAGS.output_vid, codec, fps, (width, height))
	
	frame_num = 1
	dataframe = pd.DataFrame()
	temp = pd.DataFrame()
	
	from _collections import deque
	pts = [deque(maxlen=30) for _ in range(1000)]
	print("now =", datetime.now())
	
	counter = []
	
	start_process = time.time()
	# while video is running
	
	while True:
	#for j in range(0,200):
		return_value, frame = vid.read()
		if return_value:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			image = Image.fromarray(frame)
		else:
			print('Video has ended or failed, try a different video format!')
			break
		
		print('Frame #: ', frame_num)
		frame_size = frame.shape[:2]
		image_data = cv2.resize(frame, (input_size, input_size))
		image_data = image_data / 255.
		image_data = image_data[np.newaxis, ...].astype(np.float32)
		start_time = time.time()
	
		batch_data = tf.constant(image_data)
		pred_bbox = infer(batch_data)
		for key, value in pred_bbox.items():
			boxes = value[:, :, 0:4]
			pred_conf = value[:, :, 4:]
	
		boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
			boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
			scores=tf.reshape(
				pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
				max_output_size_per_class = 50,
				max_total_size = 50,
				iou_threshold = FLAGS.iou,
				score_threshold = FLAGS.score_th
			)
	
		# convert data to numpy arrays and slice out unused elements
		num_objects = valid_detections.numpy()[0]
		bboxes = boxes.numpy()[0]
		bboxes = bboxes[0:int(num_objects)]
		scores = scores.numpy()[0]
		scores = scores[0:int(num_objects)]
		classes = classes.numpy()[0]
		classes = classes[0:int(num_objects)]
	
		# format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
		original_h, original_w, _ = frame.shape
		bboxes = utils.format_boxes(bboxes, original_h, original_w)
	
		# store all predictions in one parameter for simplicity when calling functions
		pred_bbox = [bboxes, scores, classes, num_objects]
	
		# read in all class names from config
		class_names = utils.read_class_names(cfg.YOLO.CLASSES)
	
		# by default allow all classes in .names file
		#allowed_classes = list(class_names.values())
		
		# custom allowed classes (uncomment line below to customize tracker for only people)
		allowed_classes = ['person']
	
		# loop through objects and use class index to get class name, allow only classes in allowed_classes list
		names = []
		deleted_indx = []
		for i in range(num_objects):
			class_indx = int(classes[i])
			class_name = class_names[class_indx]
			if class_name not in allowed_classes:
				deleted_indx.append(i)
			else:
				names.append(class_name)
		names = np.array(names)
		count = len(names)
	
		if FLAGS.display_count:
			cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
			print("Objects being tracked: {}".format(count))
		# delete detections that are not in allowed_classes
		bboxes = np.delete(bboxes, deleted_indx, axis=0)
		scores = np.delete(scores, deleted_indx, axis=0)
	
		# encode yolo detections and feed to tracker
		features = encoder(frame, bboxes)
		detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
	
		#initialize color map
		cmap = plt.get_cmap('tab20b')
		colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
	
		# run non-maxima supression
		boxs = np.array([d.tlwh for d in detections])
		scores = np.array([d.confidence for d in detections])
		classes = np.array([d.class_name for d in detections])
		indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
		detections = [detections[i] for i in indices]       
	
		# Call the tracker
		tracker.predict()
		tracker.update(detections)
	
		# update tracks
		for track in tracker.tracks:
			if not track.is_confirmed() or track.time_since_update > 1:
				continue 
			bbox = track.to_tlbr()
			class_name = track.get_class()
			
		# draw bbox on screen
			color = colors[int(track.track_id) % len(colors)]
			color = [i * 255 for i in color]
			cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
			cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
			cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)        
	
		# if enable info flag then print details about each track
			if FLAGS.info:
				print(f"Tracker ID: {str(track.track_id)}, Class: {class_name},  BBox Coords (xmin, ymin, xmax, ymax): {(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))}")
			temp = pd.DataFrame(
					{
						'Object':[class_name],
						'Id': [int(track.track_id)],
						'X_min': [int(bbox[0])],
						'Y_min': [int(bbox[1])],
						'X_max':[int(bbox[2])],
						'Y_max': [int(bbox[3])],
						'Frame' : [frame_num]   
					})
	
			dataframe = pd.concat([dataframe, temp],ignore_index=True)
	
			center = (int(((bbox[0])+(bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
			pts[track.track_id].append(center)
			for j in range(1, len(pts[track.track_id])):
				if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
					continue
				thickness = int(np.sqrt(64/float(j+1))*2)
				cv2.line(frame,(pts[track.track_id][j-1]),(pts[track.track_id][j]),color,thickness)
	
			height, width , _ = frame.shape
			##cv2.line(frame,(0,int(3*height/6)),(width,int(3*height/6)),(0,0,255), thickness = 2)
			#cv2.line(frame,(193,183),(650,183),(0,0,255),2)
			cv2.line(frame,startline,endline,(0,0,255),2)
			center_y = int(((bbox[1])+(bbox[3]))/2)
	
			if center_y <= int(183+height/30) and center_y >= int(183-height/30):
				if class_name == 'person':
					counter.append(int(track.track_id))
	
		total_count = len(set(counter))
	
		cv2.putText(frame,'Total Count:' + str(total_count),(0,130),0,1,(0,0,255),2)
	
		frame_num +=1
		# calculate frames per second of running detections
		fps = 1.0 / (time.time() - start_time)
		print("FPS: %.2f" % fps)
		result = np.asarray(frame)
		result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		
		#if not FLAGS.dont_show:
		#	cv2_imshow(result) # Just for colab
		#	cv2.imshow("Output Video", result)
		
		# if output flag is set, save video file
		if FLAGS.output_vid:
			out.write(result)
		if cv2.waitKey(1) & 0xFF == ord('q'): break
	
	print("Total Processing time: ",time.time()-start_process)
	cv2.destroyAllWindows()
		
	# saving the data into a csv
	dataframe.to_csv(output_csv_path, index = False)
	print("my file was successfully saved!")
	
	#upload the data to the database
	upload_to_db(output_csv_path, 'tracker')


def upload_to_db(output_csv_path, table_name):
	host = 'team-cv.cfsx82z4jthl.us-east-2.rds.amazonaws.com'
	port = 5432
	user = 'ds4a_69'
	password = 'DS4A!2020'
	database = 'postgres'
	
	# Create the engine with the db credentials
	engine=create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}', max_overflow=20)
	
	# Reading the csv file
	df = pd.read_csv(output_csv_path)
	
	# uploading the data to the database
	df.to_sql(table_name, engine, if_exists='replace', index=False, method = 'multi')
	print('Data succesfully uploaded to the database')


if __name__ == '__main__':
	try:
		app.run(main)
	except SystemExit:
		pass