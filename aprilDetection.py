from apriltag import DetectorOptions,Detector
import cv2
from time import sleep
img = cv2.imread('dmf3.png', cv2.IMREAD_COLOR)
#img = cv2.imread('img_test.png',0) #husk grayscale
cv2.imshow('image',img)
sleep(2)
"""
tag_detector_options = DetectorOptions(
        families="tag16h5",
        border=1,
        nthreads=4,
        quad_decimate=1.0,
        quad_blur=0.0,
        refine_edges=True,
        refine_decode=True,
        refine_pose=True,
        debug=False,
        quad_contours=True)

det = Detector(tag_detector_options, searchpath=["./apriltag/build/lib"])
detections, dimg = det.detect(img, return_image=True)
num_detections = len(detections)
print('Detected {} tags'.format(num_detections))
"""

