import numpy as np
import cv2
import subprocess as sp

command = [ 'ffmpeg',
            '-i', 'myHolidays.mp4',
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-']

pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)

# read 420*360*3 bytes (= 1 frame)
raw_image = pipe.stdout.read(420*360*3)
# transform the byte read into a numpy array
image =  numpy.fromstring(raw_image, dtype='uint8')
image = image.reshape((360,420,3))
# throw away the data in the pipe's buffer.
pipe.stdout.flush()

# cap = cv2.VideoCapture("/home/mirl/egibbons/data/video_data/Hollywood2/AVIClips/actioncliptest00681.avi")

# while not cap.isOpened():
#     print("Error in opening")


# while(True):
#     ret, frame = cap.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     cv2.imshow("frame", gray)
    

# cap.release()
# cv2.destroyAllWindows()
