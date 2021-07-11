"""
This program takes a live videofeed from the Lepton and draws outlines around the detected person.

You may have to change the cameraID depending on your setup.
"""

import cv2
import numpy as np

cameraID = 1

#Get the videofeed
videoCapture = cv2.VideoCapture(cameraID)

if videoCapture.isOpened():  #try to get the first frame in the video feed
    rval, frame = videoCapture.read()
else:
    rval = False

while rval:     #repeatedly update the frame
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  #convert frame to hsv
    frame_v = frame_hsv[:,:,2]  #set V value to 2
    # cv2.imshow("HSV Frame", frame_v)

    blurredBrightness = cv2.bilateralFilter(frame_v,9,150,150)
    thresh = 50     #threshold for counting as an edge
    edges = cv2.Canny(blurredBrightness,thresh,thresh*2, L2gradient=True)   #edge detection

    _,mask = cv2.threshold(blurredBrightness,200,1,cv2.THRESH_BINARY)
    erodeSize = 5  #how much to diminish features in the image
    dilateSize = 9  #makes remaining features more prominent
    eroded = cv2.erode(mask, np.ones((erodeSize, erodeSize)))
    mask = cv2.dilate(eroded, np.ones((dilateSize, dilateSize)))

    edges1 = cv2.resize(cv2.cvtColor(mask*edges, cv2.COLOR_GRAY2RGB), (640, 480))
    frame_with_edges = cv2.resize(cv2.cvtColor(mask*edges, cv2.COLOR_GRAY2RGB) | frame, (640, 480))     #the | character overlays the 2 frames
    # frame_with_edges = cv2.resize(mask*edges | frame_v, (640, 480))     #the | character overlays the 2 frames

    #Show both the edges overlaid on the original image, and just the edges
    cv2.imshow("Edges", edges1)
    cv2.imshow("Edges Overlaid", frame_with_edges)

    rval, frame = videoCapture.read()

    key = cv2.waitKey(20)
    if key == 27:   #exit on ESC
        break
