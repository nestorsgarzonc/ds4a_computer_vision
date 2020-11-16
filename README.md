# yolov4-deepsort
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

Person tracking implemented with YOLOv4, DeepSort, and TensorFlow. 

### You can find the frontend repository of the application [here](https://github.com/nestorsgarzonc/Off_Corss_Front_End)

### you can see the final deployed web application [here](http://ec2-18-217-12-2.us-east-2.compute.amazonaws.com:8050/)

## Command Line Args Reference

```bash
 person_tracker.py:
  --output_vid: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
    (default: None)
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'XVID')
  --weights_path: path to weights file
    (default: './checkpoints/yolov4-416')
  --size: resize images to
    (default: 416)
  --iou: iou threshold
    (default: 0.45)
  --score_th: confidence threshold
    (default: 0.50)
  --info: print detailed info about tracked objects
    (default: False)
  --display_count: count objects being tracked on screen
    (default: False)
  --output_csv_path: path to output detections csv file
    (default: './outputs/video_data.csv')
  --count_csv_path: path to output count csv file
    (default: './outputs/count_data.csv')
  --videos_repository_path: path to the stored videos
    (default: '/content/drive/My Drive/Data Vision Artificial/San Diego/Videos')
```

### References  

  * [yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort)