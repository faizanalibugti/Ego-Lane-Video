import cv2
from scipy.misc import imresize
import numpy as np
from grabscreen import grab_screen
import time
from keras.models import load_model
from moviepy.editor import *

# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """
    print(image.shape)
    # Get image ready for feeding into model
    small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = imresize(lane_drawn, (626, 1000, 3))

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result


if __name__ == '__main__':
    # Load Keras model
    model = load_model('full_CNN_model.h5')
    # Create lanes object
    lanes = Lanes()
    # while (True):
    #     last_time = time.time()
    #     screen = grab_screen(region=(0, 40, 1000, 600))
    #     capture = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)

    #     output = road_lines(capture)

    #     cv2.imshow('Output', output)

    #     #cv2.imshow('window', screen)
    #     print("fps: {}".format(1 / (time.time() - last_time)))

    #     if cv2.waitKey(25) & 0xFF == ord("q"):
    #         cv2.destroyAllWindows()
    #         break

    white_output = "output.mp4"
    clip1 = VideoFileClip("drive.mp4")
    white_clip = clip1.fl_image(road_lines)
    white_clip.write_videofile(white_output, audio=False)