import cv2
import math
import time
import numpy as np
import configparser
from utils.apriltag import Detector, DetectorOptions
from utils.api import API
from classes.drop import Drop

# Initialize
config = configparser.ConfigParser()
config.read('config.ini')
start = time.time()
bg = None
fpslist = []
dropThres = float(config['DEFAULT']['droplet_threshold'])
numDroplets = 0

# If recorded video:
cap = cv2.VideoCapture('videos/open/data_38.h264')


class bcolors:
    HEADER = '\033[95m'
    WARNING = '\u001b[32m'
    ENDC = '\033[0m'


print(f"{bcolors.WARNING}Starting digital microfluidics computer vision algorithm {bcolors.ENDC}")
time.sleep(0.3)
print(f"{bcolors.WARNING}3.. {bcolors.ENDC}")
time.sleep(0.1)
print(f"{bcolors.WARNING}2..{bcolors.ENDC}")
time.sleep(0.1)
print(f"{bcolors.WARNING}1.. {bcolors.ENDC}")
time.sleep(0.2)


drops = []


# Function to add droplets into droplet list
def addDrop(drop):
    exist = True
    # Check if there exist any droplets
    if len(drops) == 0:
        drops.append(drop)
    # Compare already detected droplets to new droplet
    for droplet in drops:
        if np.linalg.norm(np.subtract(droplet.center, drop.center)) < 3:
            droplet.center = drop.center
            droplet.radius = drop.radius
            droplet.history.append(center)
            droplet.acc = abs(((1/500)*(int(len(droplet.history)) * droplet.circularity)))
            exist = True
            break
        else:
            exist = False
    if not(exist):
        drops.append(drop)


# Function to calculate color of droplet
def color(center, radius):
    list = []
    r = []
    g = []
    b = []
    x0 = center[0]
    y0 = center[1]
    for i in range(0, radius, 2):
        list.append(frame[x0 + i, y0].tolist())
        list.append(frame[x0 - i, y0].tolist())
        list.append(frame[x0, y0 + i].tolist())
        list.append(frame[x0, y0 - i].tolist())
    for col in list:
        r.append(col[0])
        g.append(col[1])
        b.append(col[2])
    # Average of central pixels in the horizontal and vertical axis
    rgb = [int(sum(r) / len(r)), int(sum(g) / len(g)), int(sum(b) / len(b))]
    return rgb


# Iterate over every frame
while True:
    curDroplets = 0
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (int(config['DEFAULT']['Gaussian_blur_kernel']), int(config['DEFAULT']['Gaussian_blur_kernel'])), 0)

    # First iteration of stream processing
    if bg is None:
        # April Tag cropping functions from Luca Pezzarossa
        # Creating a detector option object and creating a detector
        tag_detector_options = DetectorOptions(
            families="tag16h5",
            border=int(config['DEFAULT']['border']),
            nthreads=int(config['DEFAULT']['nthreads']),
            quad_decimate=float(config['DEFAULT']['quad_decimate']),
            quad_blur=float(config['DEFAULT']['quad_blur']),
            refine_edges=True,
            refine_decode=True,
            refine_pose=True,
            debug=False,
            quad_contours=True)
        det = Detector(tag_detector_options, searchpath=["./apriltag/build/lib"])

        detections, dimg = det.detect(gray, return_image=True)

        # If less than 4 April tags detected program terminates
        # if len(detections) < 4:
        #     print("  Detected less than 3 apriltags. Terminating.")
        #     break

        # # Selecting the 4 apriltags candidates in the corners
        # selected_detections = []
        # for i, detection in enumerate(detections):
        #     decision_margin = detection.decision_margin
        #     if decision_margin >= config.minimum_decision_margin:
        #         selected_detections.append(detection)
        #
        # # If less than 4 April tags is selected program terminates
        # if len(selected_detections) < 4:
        #     print("  Selected less than 4 apriltags. Terminating.")
        #     break
        #
        # # Find corners to crop frame
        # for tag in selected_detections:
        #     if tag.tag_id == 0:
        #         x_crop1 = int(tag.center[0]+12)
        #         y_crop1 = int(tag.center[1]+12)
        #     if tag.tag_id == 4:
        #         x_crop2 = int(tag.center[0] - 12)
        #         y_crop2 = int(tag.center[1] - 12)
        #
        # gray = frame[y_crop1:y_crop2, x_crop1:x_crop1]

        # Reference frame for movement detection
        bg = gray
        continue

    # Cropping each frame
    # gray = frame[y_crop1:y_crop2, x_crop1:x_crop2]

    # Movement detection
    diff_frame = cv2.absdiff(bg, gray)
    thresh_frame = cv2.threshold(diff_frame, int(config['DEFAULT']['difference_threshold']), 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
    cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cen = 0
    # Loop over each detected contour
    for contour in cnts:
        motion = 1
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        electrode_center = (int(round((int(x)-36)/8)), int(round((int(y)-8)/8)))
        radius = int(radius)
        circumference = cv2.arcLength(contour, True)
        circularity = circumference ** 2 / (4 * math.pi * (radius * math.pi ** 2))
        contour = cv2.convexHull(contour)
        if int(config['DEFAULT']['droplet_radius_max']) > radius > int(config['DEFAULT']['droplet_radius_min']) and circularity < int(config['DEFAULT']['circularity_min']):
            # col = color(center, radius)
            drop = Drop(electrode_center, radius, 1, circularity)
            curDroplets += 1
            addDrop(drop)
            # Visualisation by creating a green circle around droplet and text showing accuracy of the droplet
            cv2.circle(thresh_frame, center, radius, (0, 255, 0), 2)
            for drop in drops:
                if drop.acc > dropThres:
                    numDroplets += 1
                if center == drop.center:
                    cv2.putText(frame, str(round(drop.acc, 3)), (center[0], center[1] + (radius * 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Circle Edge Detection and Circle Hough Transform
    if numDroplets > curDroplets:
        # Canny Edge detection
        can = cv2.Canny(gray, int(config['DEFAULT']['canny_param1']), int(config['DEFAULT']['canny_param2']))
        # Circle Hough transform
        detected_circles = cv2.HoughCircles(can,
                                            cv2.HOUGH_GRADIENT, 1, 20, param1=int(config['DEFAULT']['circle_edge_param1']),
                                            param2=int(config['DEFAULT']['circle_edge_param2']), minRadius=int(config['DEFAULT']['droplet_radius_min']), maxRadius=int(config['DEFAULT']['droplet_radius_max']))

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        # Loop over each detected circle
        for dat in detected_circles[0, :]:
            a, b, radius = dat[0], dat[1], dat[2]
            center = (a,b)
            col = color(center, radius)
            electrode_center = (int(round((int(a) - 36) / 8)), int(round((int(b) - 8) / 8)))
            drop = Drop(electrode_center, radius, col, circularity)
            # Draw the circumference of the circle.
            cv2.circle(frame, (a, b), radius, (0, 255, 0), 2)

    print(drops)
    """
    # Show fps and memory
    print("FPS: ", 1.0 / (time.time() - start_time))
    fpslist.append(int(1.0 / (time.time() - start_time)))
    fpslist.sort()
    print(fpslist)
    print(sum(fpslist)/len(fpslist))
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    """
    if int(config['DEFAULT']['show_result']) == 1:
        cv2.imshow("Droplets Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
