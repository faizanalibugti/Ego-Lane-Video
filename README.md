# Ego Lane Segmentation on Video

This code will allow inference of ego lane segmentation model on video data

You must have moviepy library installed 

1. Download or clone this repository
2. Rename your video as **drive.mp4**
2. In Anaconda Prompt run **python egolane.py**

It may prompt with the error:

cv2.error: OpenCV(3.4.2) C:\projects\opencv-python\opencv\modules\core\src\arithm.cpp:659: error: (-209:Sizes of input arguments do not match) The operation is neither 'array op array' (where arrays have the same size and the same number of channels), nor 'array op scalar', nor 'scalar op array' in function 'cv::arithm_op'

To resolve this modify **Line 45** of egolane.py to input image resolution of video

You will find correct resolution on 4th line of where you ran the command on Anaconda Prompt

Final output will be output.mp4