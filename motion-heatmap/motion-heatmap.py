# pip install opencv-contrib-python
import numpy as np
import cv2
import copy

def heatmap_video(path_in, path_out, frames_sec = 2, thresh = 2, maxValue = 3):
    """
    Function to generate heatmap in the store
    path_in: path with the location of the video to process
    path_out:  path where the imagen results will be save
    frames_sec: number of frames per second to process
    thresh: apply a binary threshold only keeping pixels above thresh and setting the result to maxValue.
            if you want motion to be picked up more, increase the value of maxValue
    maxValue: to pick up the least amount of motion over time, set maxValue = 1
    """

    cap = cv2.VideoCapture(path_in)
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) # frames per second
    duration = round(num_frames / fps, 2) # duration of the video in seconds
    print('Total number of frames to process: {}'.format(num_frames))
    print('Duration of the video in seconds: {}'.format(duration))
    step = round(fps / frames_sec)

    first_iteration_indicator = 1
    for i in range(0, num_frames, step):

        if (first_iteration_indicator == 1):
            ret, frame = cap.read()
            first_frame = copy.deepcopy(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0

        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            print('Frame process... ' + str(i) + ' of ' + str(num_frames))
            print('Second process... ' + str(int(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000) + ' of ' + str(duration))
            print('...')
            ret, frame = cap.read()  # read a frame

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale

            fgmask = fgbg.apply(gray)  # remove the background
            ret, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY)

            # add to the accumulated image
            accum_image = cv2.add(accum_image, th1)

    # apply a color map
    # COLORMAP_PINK also works well, COLORMAP_BONE is acceptable if the background is dark
    color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)

    # overlay the color mapped image to the first frame
    result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)

    # save the final overlay image
    cv2.imwrite(path_out, result_overlay)

    # cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    heatmap_video('Hermeco Oficinas_10.50.60.47_1_20201001161330_20201001161924_1601609295690.mp4',
                  'diff-overlay.jpg', frames_sec = 2, thresh = 4, maxValue = 2)