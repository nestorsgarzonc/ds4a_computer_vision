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
from utils import *
import matplotlib.path as mpltPath
from datetime import timedelta
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
flags.DEFINE_string('output_csv_path', './outputs/video_data.csv','path to output detections csv file')
flags.DEFINE_string('startline', '614,95','start point of the line for the entrance of the store')
flags.DEFINE_string('endline', '807,95','end point of the line for the entrance of the store')
flags.DEFINE_string('count_csv_path', './outputs/count_data.csv','path to output count csv file')

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
	# STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
	input_size = FLAGS.size
	video_path = FLAGS.video_path
	startline = tuple(map(int,FLAGS.startline.split(',')))
	endline = tuple(map(int,FLAGS.endline.split(',')))
	output_csv_path = FLAGS.output_csv_path
	count_csv_path = FLAGS.count_csv_path

	# Loading the stores configuration JSON file
	stores_config_filename = 'stores_sections.json'
	stores_config_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), stores_config_filename)
	stores_sections = load_json_file(stores_config_filepath)
		
	#Extract the date and time information from the video_path string
	if len(re.findall('[0-9]{14}',video_path)) == 2:
		time_start_vid, time_end_vid = re.findall('[0-9]{14}',video_path)
		time_start_vid_dt = datetime.strptime(str(time_start_vid), '%Y%m%d%H%M%S')
		time_end_vid_dt = datetime.strptime(str(time_end_vid), '%Y%m%d%H%M%S')
		camera = int(re.findall(r'_([0-9]{1})_', video_path.lower())[0])
		
	# Extract the name of the store from the video_path string
	store_name = re.findall(r'/([a-z0-9\s]*)_*',video_path.lower())[-1]
	if store_name == 'hermeco oficinas':
		store_name = 'san diego'

	saved_model_loaded = tf.saved_model.load(FLAGS.weights_path, tags=[tag_constants.SERVING])
	infer = saved_model_loaded.signatures['serving_default']
	
	# begin video capture
	try:
		vid = cv2.VideoCapture(int(video_path))
	except:
		vid = cv2.VideoCapture(video_path)
	
	
	
	# by default VideoCapture returns float instead of int
	out = None
	width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(vid.get(cv2.CAP_PROP_FPS))
	frame_count  = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of frames in the video
	codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
	delta_time = (time_end_vid_dt -time_start_vid_dt)/frame_count
	
	# get video ready to save locally if flag is set
	if FLAGS.output_vid:		
		out = cv2.VideoWriter(FLAGS.output_vid, codec, fps, (width, height))
	
	frame_num = 1
	detections_df = pd.DataFrame(
		{
			'Store_name': [],
			'Start_date': [],
			'End_date': [],
			'current_datetime':[],
			'Camera': [],
			'Object':[],
			'Id': [],
			'X_center_original': [],
			'Y_center_original': [],
			'X_center_perspective': [],
			'Y_center_perspective': [],
			'X_min': [],
			'Y_min': [],
			'X_max':[],
			'Y_max': [],
			'Frame' : []   
		})
	temp = pd.DataFrame()
	
	from _collections import deque
	pts = [deque(maxlen=15) for _ in range(1000)]
	print("now =", datetime.now())
	
	counter = []
	
	start_process = time.time()
	
	# while video is running
	while True:
	#for j in range(0,1000):
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

		# Computing the time has passed since the beggining of the video
		delta_time_frame = delta_time*(frame_num-1)

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
					
			center = (int(((bbox[0])+(bbox[2]))/2), int(bbox[3]))

			# Loop for the diferent configured sections of the store for the current camera
			for sec in stores_sections[store_name][f'camera_{camera}'].keys():
				# Get the 4 points of the section
				bound_section_points = 	stores_sections[store_name][f'camera_{camera}'][sec]['camera_view_points']
				# Verify if the center point of the detection is in the section region
				mpltPath_path = mpltPath.Path(bound_section_points)
				inside = mpltPath_path.contains_point(list(center))
				if inside:
					# Get the perspective transformation matrix
					transform_matrix = np.array(stores_sections[store_name][f'camera_{camera}'][sec]['transformation_matrix'])
					# Apply the transformation matrix to transform the point to the blueprint perspective
					transformed_center = point_perspective_transform(center, transform_matrix)[0]
				else:
					transformed_center = [0,0]

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
			
			temp = pd.DataFrame(
					{
						'Store_name': [store_name],
						'Start_date': [time_start_vid_dt],
						'End_date': [time_end_vid_dt],
						'current_datetime':[time_start_vid_dt + delta_time_frame],
						'Camera': [int(camera)],
						'Object':[class_name],
						'Id': [int(track.track_id)],
						'X_center_original': [int(center[0])],
						'Y_center_original': [int(center[1])],
						'X_center_perspective': [int(transformed_center[0])],
						'Y_center_perspective': [int(transformed_center[1])],
						'X_min': [int(bbox[0])],
						'Y_min': [int(bbox[1])],
						'X_max':[int(bbox[2])],
						'Y_max': [int(bbox[3])],
						'Frame' : [int(frame_num)]   
					})	
			detections_df = pd.concat([detections_df, temp],ignore_index=True)

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

	for f in range(1,frame_num):
		if f not in detections_df.Frame.to_list():
			# Computing the time has passed since the beggining of the video
			delta_time_frame = delta_time*(f-1)
			# default row in the dataframe if there are 0 detections in the frame
			temp = pd.DataFrame(
					{
						'Store_name': [store_name],
						'Start_date': [time_start_vid_dt],
						'End_date': [time_end_vid_dt],
						'current_datetime':[time_start_vid_dt + delta_time_frame],
						'Camera': [camera],
						'Object':['default'],
						'Id': [int(0)],
						'X_center_original': [int(0)],
						'Y_center_original': [int(0)],
						'X_center_perspective': [int(0)],
						'Y_center_perspective': [int(0)],
						'X_min': [int(0)],
						'Y_min': [int(0)],
						'X_max':[int(0)],
						'Y_max': [int(0)],
						'Frame' : [int(f)]   
					})	
			detections_df = pd.concat([detections_df, temp],ignore_index=True)
	detections_df = detections_df.sort_values(by= 'Frame')

	print("Total Processing time: ",time.time()-start_process)
	cv2.destroyAllWindows()
		
	# saving the detections data into a csv
	detections_df.to_csv(output_csv_path, index = False)
	print("The detections file was successfully saved!")
	
	# saving the count data into into a csv
	count_df = pd.DataFrame(
		{
			'Store_name': [store_name],
			'Start_date': [time_start_vid_dt],
			'End_date': [time_end_vid_dt],
			'Camera': [camera],
			'Count': [total_count]
		}
	)
	count_df.to_csv(count_csv_path, index = False)
	print("The counts file was successfully saved!")
	
	#upload the detections data to the database
	#upload_to_db(output_csv_path, 'tracker') # passing the csv path
	upload_to_db(detections_df, 'tracker') # passing the dataframe
	
	#upload the count data to the database
	upload_to_db(count_df, 'counts')


if __name__ == '__main__':
	try:
		app.run(main)
	except SystemExit:
		pass