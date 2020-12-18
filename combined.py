import cv2
import math
import time
import numpy as np
import configparser
import json
import resource

# Initialize
config = configparser.ConfigParser()
config.read('config2.ini')
start = time.time()
cap = cv2.VideoCapture('data_38.h264')
bg = None


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
fpslist = []
# Drop object class
class Drop:
    def __init__(self, center, radius, circularity):
        self.center = center
        self.radius = radius
        self.color = color
        self.circularity = circularity
        self.history = []
        self.acc = 0

    def __repr__(self):
        return ("( center = " + str(self.center) + ", radius = " + str(self.radius) + ", average color = "+str(self.color)+", circularity = "+str(round(self.circularity,3))+", length of history = "+str(len(self.history))+", accuracy = "+str((self.acc))+" ) ")

    def __str__(self):
        return ("( center = " + str(self.center) + ", radius = " + str(self.radius) + ", average color = "+str(self.color)+", circularity = "+str(round(self.circularity,3))+", length of history = "+str(len(self.history))+", accuracy = "+str((self.acc))+" ) ")

    def dump(self):
        return ("( center = " + str(self.center) + ", radius = " + str(self.radius) + ", average color = "+str(self.color)+", circularity = "+str(round(self.circularity,3))+", length of history = "+str(len(self.history))+", accuracy = "+str((self.acc))+" ) ")


drops = []


def addDrop(drop):
    exist = True
    if len(drops) == 0:
        drops.append(drop)
    for droplet in drops:
        if np.linalg.norm(np.subtract(droplet.center,drop.center)) < 50:
            droplet.center = drop.center
            droplet.radius = drop.radius
            droplet.history.append(center)
            #print(np.log((int(len(droplet.history))/200)*(droplet.circularity)))
            #droplet.acc = abs((int(len(droplet.history))/500)*(1/droplet.circularity))
            droplet.acc = abs(((1/500)*(int(len(droplet.history)) * droplet.circularity)))

            exist = True
            break
        else:
            exist = False
    if not(exist):
        drops.append(drop)

def color(center,radius):
    list = []
    r = []
    g = []
    b = []
    x0 = center[0]
    y0 = center[1]
    for i in range(0,radius,2):
        list.append(frame[x0 + i, y0].tolist())
        list.append(frame[x0 - i, y0].tolist())
        list.append(frame[x0, y0 + i].tolist())
        list.append(frame[x0, y0 - i].tolist())
    for col in list:
        r.append(col[0])
        g.append(col[1])
        b.append(col[2])
    rgb = [int(sum(r) / len(r)), int(sum(g) / len(g)), int(sum(b) / len(b))]
    return rgb

def API():
    jayson = json.dumps([drop.dump() for drop in drops])
    return jayson

while True:
    start_time = time.time()
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if time.time() - start > 1000 or bg is None:
        bg = gray
        start = time.time()
        continue

    diff_frame = cv2.absdiff(bg, gray)

    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
    cv2.imshow("Diff Frame", thresh_frame)
    cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cen = 0
    for contour in cnts:
        motion = 1
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        circumference = cv2.arcLength(contour, True)
        circularity = circumference ** 2 / (4 * math.pi * (radius * math.pi ** 2))
        contour = cv2.convexHull(contour)
        """"""

        if int(config['DEFAULT']['droplet_radius_max']) > radius > int(config['DEFAULT']['droplet_radius_min']) and circularity < int(config['DEFAULT']['circularity_min']) and center[0] > 200 and center[1] > 100 and center[0] < 380 and center[1] < 330:
            #col = color(center,radius)
            drop = Drop(center, radius, circularity)
            addDrop(drop)
            cv2.circle(thresh_frame, center, radius, (0, 255, 0), 2)
            #cv2.imshow("Droplets Frame", thresh_frame)
            for drop in drops:
                if center == drop.center:
                    cv2.putText(frame, str(round(drop.acc, 3)), (center[0], center[1] + (radius * 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)



            """
                if center == drop.center:
                    if drop.acc>0.99:
                        cv2.putText(frame, str(round(0.8+(np.random.randn()/10),3)), (center[0], center[1] + (radius * 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        cv2.putText(frame, str(round(drop.acc, 3)), (center[0], center[1] + (radius * 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
"""

    print(drops)
    """
    #print("FPS: ", 1.0 / (time.time() - start_time))
    fpslist.append(int(1.0 / (time.time() - start_time)))
    fpslist.sort()
    print(fpslist)
    print(sum(fpslist)/len(fpslist))
    
    if ((time.time() - start_time)>1):
        time.sleep(2)
    
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    """
    cv2.imshow("Droplets Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



