import cv2
import numpy as np
from utils import *
import re
import matplotlib.path as mpltPath
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_integer('sections', 2, 'number of section to divide the store')
flags.DEFINE_string('video_path', 'D:/Giovanny/DS4A/Project/Videos_convertidos/Hermeco Oficinas_10.50.60.47_4_20201003172927_20201003173355_1602791609944.mp4', 'path to input video')
flags.DEFINE_string('filename', 'stores_sections.json', 'JSON filename')
flags.DEFINE_string('base_image', 'D:/Giovanny/DS4A/Project/Planos/Planos_San_Diego_1.jpg', 'Blue print of the store')

def main(_argv):
    global counter, sec_counter, base_points_count, base_control

    sections = FLAGS.sections # Number of sections the store are going to be splitted
    base_points_count = 0
    base_points = np.zeros((sections,4,2), dtype = int) # Array to store the base points selected on the image
    counter = 0 # To count the number of clicks
    test_points = [] # List to store the clicks after the 4 point region is defined    
    video_path = FLAGS.video_path
    sec_counter = 0 # To count the number of the current section
    # Defining the variable to store the 4 clicks for each section, each click has x and y coordinates
    circles = np.zeros((sections,4,2), dtype=int) # Array to store the camera view points selected on the image
    base_control = True # Variable to indicate if we are selecting points in the blueprint or in the camera image

    # Mouse click callback
    def mousePoints(event, x, y, flags, params):
        global counter, sec_counter, base_points_count, base_control

        if event == cv2.EVENT_LBUTTONDOWN:  
            # logic to store four points at a time of the blueprint view and the camera view
            if base_control:                
                if sec_counter < sections:
                    if base_points_count < 4:
                        base_points[sec_counter][base_points_count] = x,y
                        base_points_count += 1

                    if base_points_count == 4:
                        base_control = False
                        base_points_count = 0
            else:
                if sec_counter == sections:
                    # if the test point is inside the sections, it will be stored in the list and
                    # draw in the camera image
                    mpltPath_path = mpltPath.Path([circles[sec][0],circles[sec][1],circles[sec][2],circles[sec][3]])
                    inside = mpltPath_path.contains_point([x,y])
                    if inside:                        
                        test_points.append([x,y])
                        cv2.circle(img_copy, (x, y), 4, (255,255,0), cv2.FILLED) 
                        print(test_points) 
                if sec_counter < sections:            
                    if counter < 4:
                        circles[sec_counter][counter] = x,y
                        counter += 1
                    if counter == 4:
                        counter = 0
                        sec_counter += 1 
                        if sec_counter < sections:
                            base_control = True
            print('sec_counter: '+ str(sec_counter))
            print('base_point_count: ' + str(base_points_count))
            print('counter: ' + str(counter))
            print('base_control: ' + str(base_control))

    # Get the frame of the store camera that we are going to configure
    vid = cv2.VideoCapture(video_path)
    return_value, img = vid.read()
    img_copy = img.copy()
    base_image_path = FLAGS.base_image
    base_image = cv2.imread(base_image_path)
    base_image_copy = base_image.copy()    

    #Extract the name of the store and camera number from the video_path string
    store_name = re.findall(r'/([a-z0-9\s]*)_*',video_path.lower())[-1]
    camera = int(re.findall(r'_([0-9]{1})_', video_path.lower())[0])
    if store_name == 'hermeco oficinas':
	    store_name = 'san diego'

    while True:
        if not base_control:            
            if sec_counter == sections:                
                for sec in range(0, sections):
                    # Get the image size
                    base_height, base_width = base_image.shape[:2]

                    # Compute an array  of the points selected in both image for each section
                    pts1 = np.float32([circles[sec][0],circles[sec][1],circles[sec][2],circles[sec][3]]) # camera image points
                    pts2 = np.float32([base_points[sec][0],base_points[sec][1],base_points[sec][2],base_points[sec][3]])# 

                    # Compute the transformation matrix
                    matrix = cv2.getPerspectiveTransform(pts1, pts2)

                    # warp the camera image with the perspective transformation
                    warped_img = cv2.warpPerspective(img, matrix, (base_width,base_height))

                    # transformed test points 
                    points_p = point_perspective_transform(test_points,matrix) # single point perspective

                    for i in range(0,len(points_p)):
                        cv2.circle(warped_img, (int(points_p[i][0]), int(points_p[i][1])), 4, (0,0,255), cv2.FILLED)
                    
                    #cv2.imshow('Warped Image', warped_img)

                    # Create a mask
                    mask = np.zeros(base_image.shape, dtype=np.uint8)
                    roi_corners = np.int32(base_points[sec])               
                    
                    # Fill in the region selected with white color
                    filled_mask = mask.copy()
                    cv2.fillConvexPoly(filled_mask, roi_corners, (255, 255, 255))

                    # Invert the mask color
                    inverted_mask = cv2.bitwise_not(filled_mask)

                    # Bitwise AND the mask with the base image
                    masked_image = cv2.bitwise_and(base_image, inverted_mask)

                    # # Using Bitwise OR to merge the two images
                    output = cv2.bitwise_or(warped_img, masked_image)
                    cv2.imshow('Fused Image', output)                    

            # draw the circles for each of the points selected on the camera image
            for sec in range(0, sections):
                for x in range(0,4):
                    cv2.circle(img_copy, (circles[sec][x][0], circles[sec][x][1]), 4, (255,0,0), cv2.FILLED)

            # Display the camera image to select the four points with clicks over the image                  
            cv2.destroyWindow('Base Image')  
            cv2.imshow('Camera Image', img_copy)                  
            cv2.setMouseCallback('Camera Image', mousePoints)
        else:         
            # draw the circles for each of the points selected on the blueprint image
            for sec in range(0,sections):
                for x in range(0,4):
                    cv2.circle(base_image_copy, (base_points[sec][x][0], base_points[sec][x][1]), 4, (255,0,0), cv2.FILLED)

            # Display the blueprint image to select the four points with clicks over the image                  
            cv2.destroyWindow('Camera Image')    
            cv2.imshow('Base Image', base_image_copy)
            cv2.setMouseCallback('Base Image', mousePoints)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):       
            break

    # Filename of the JSON file
    filename = FLAGS.filename   

    try:
        # Loading the saved dictionary of the sections information to overwrite or create new information
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        store_sections = load_json_file(file_path)
    except:
        store_sections = dict() # dictionary to store the sections information    
        store_sections[store_name] = dict()

    # Creating the subdivisions of the JSON file that stores the configurations
    store_sections[store_name][f'camera_{camera}'] = dict()
    for sec in range(0,sections):
        store_sections[store_name][f'camera_{camera}'][f'section_{sec+1}'] = dict()
        store_sections[store_name][f'camera_{camera}'][f'section_{sec+1}'][f'base_view_points'] = []
        store_sections[store_name][f'camera_{camera}'][f'section_{sec+1}'][f'camera_view_points'] = []
        store_sections[store_name][f'camera_{camera}'][f'section_{sec+1}'][f'transformation_matrix'] = matrix.tolist()
        for i in range(0,len(circles[sec])):
            store_sections[store_name][f'camera_{camera}'][f'section_{sec+1}'][f'base_view_points'].append([int(base_points[sec][i][0]), int(base_points[sec][i][1])])
            store_sections[store_name][f'camera_{camera}'][f'section_{sec+1}'][f'camera_view_points'].append([int(circles[sec][i][0]), int(circles[sec][i][1])])

    # Saving the dictionary to a json file
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    save_json_file(store_sections, output_path)

if __name__ == '__main__':
	try:
		app.run(main)
	except SystemExit:
		pass