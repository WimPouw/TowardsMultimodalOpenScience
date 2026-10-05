# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:40:15 2022

@author: Wim Pouw
"""

#load in required packages
import mediapipe as mp #mediapipe
import cv2 #opencv
import math #basic operations
import numpy as np #basic operations
import pandas as pd #data wrangling
import csv #csv saving
import os #some basic functions for inspecting folder structure etc.

#list all videos in input_videofolder
from os import listdir
from os.path import isfile, join
mypath = "./Input_Videos/" #this is your folder with (all) your video(s)
vfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] #loop through the filenames and collect them in a list
#time series output folder
outputf_mask = "./Output_MaskedVideos/"
outtputf_ts = "./Output_TimeSeries/"

#check videos to be processed
print("The following folder is set as the output folder where all the pose time series are stored")
print(os.path.abspath(outtputf_ts))
print("\n The following folder is set as the output folder for saving the masked videos ")
print(os.path.abspath(outputf_mask))
print("\n The following video(s) will be processed for masking: ")
print(vfiles)


#initialize modules and functions

#load in mediapipe modules
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holistic_landmarker.task")
if not os.path.isfile(model_path):
    raise FileNotFoundError(
        model_path + " not found. "
        "Download https://storage.googleapis.com/mediapipe-assets/holistic_landmarker.task "
        "and place it beside Masked-PiperPY.py."
    )
# Import drawing_utils and drawing_styles.
mp_drawing = vision.drawing_utils
hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
face_connections = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
pose_connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

##################FUNCTIONS AND OTHER VARIABLES
#landmarks 33x that are used by Mediapipe (Blazepose)
markersbody = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
          'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 
          'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX',
          'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
          'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

markershands = ['LEFT_WRIST', 'LEFT_THUMB_CMC', 'LEFT_THUMB_MCP', 'LEFT_THUMB_IP', 'LEFT_THUMB_TIP', 'LEFT_INDEX_FINGER_MCP',
              'LEFT_INDEX_FINGER_PIP', 'LEFT_INDEX_FINGER_DIP', 'LEFT_INDEX_FINGER_TIP', 'LEFT_MIDDLE_FINGER_MCP', 
               'LEFT_MIDDLE_FINGER_PIP', 'LEFT_MIDDLE_FINGER_DIP', 'LEFT_MIDDLE_FINGER_TIP', 'LEFT_RING_FINGER_MCP', 
               'LEFT_RING_FINGER_PIP', 'LEFT_RING_FINGER_DIP', 'LEFT_RING_FINGER_TIP', 'LEFT_PINKY_FINGER_MCP', 
               'LEFT_PINKY_FINGER_PIP', 'LEFT_PINKY_FINGER_DIP', 'LEFT_PINKY_FINGER_TIP',
              'RIGHT_WRIST', 'RIGHT_THUMB_CMC', 'RIGHT_THUMB_MCP', 'RIGHT_THUMB_IP', 'RIGHT_THUMB_TIP', 'RIGHT_INDEX_FINGER_MCP',
              'RIGHT_INDEX_FINGER_PIP', 'RIGHT_INDEX_FINGER_DIP', 'RIGHT_INDEX_FINGER_TIP', 'RIGHT_MIDDLE_FINGER_MCP', 
               'RIGHT_MIDDLE_FINGER_PIP', 'RIGHT_MIDDLE_FINGER_DIP', 'RIGHT_MIDDLE_FINGER_TIP', 'RIGHT_RING_FINGER_MCP', 
               'RIGHT_RING_FINGER_PIP', 'RIGHT_RING_FINGER_DIP', 'RIGHT_RING_FINGER_TIP', 'RIGHT_PINKY_FINGER_MCP', 
               'RIGHT_PINKY_FINGER_PIP', 'RIGHT_PINKY_FINGER_DIP', 'RIGHT_PINKY_FINGER_TIP']
facemarks = [str(x) for x in range(478)] #there are 478 points for the face mesh (see google holistic face mesh info for landmarks)

print("Note that we have the following number of pose keypoints for markers body")
print(len(markersbody))

print("\n Note that we have the following number of pose keypoints for markers hands")
print(len(markershands))

print("\n Note that we have the following number of pose keypoints for markers face")
print(len(facemarks ))

#set up the column names and objects for the time series data (add time as the first variable)
markerxyzbody = ['time']
markerxyzhands = ['time']
markerxyzface = ['time']

for mark in markersbody:
    for pos in ['X', 'Y', 'Z', 'visibility']: #for markers of the body you also have a visibility reliability score
        nm = pos + "_" + mark
        markerxyzbody.append(nm)
for mark in markershands:
    for pos in ['X', 'Y', 'Z']:
        nm = pos + "_" + mark
        markerxyzhands.append(nm)
for mark in facemarks:
    for pos in ['X', 'Y', 'Z']:
        nm = pos + "_" + mark
        markerxyzface.append(nm)

#check if there are numbers in a string
def num_there(s):
    return any(i.isdigit() for i in s)

#take some google classification object and convert it into a string
def makegoginto_str(gogobj):
    gogobj = str(gogobj).strip("[]")
    gogobj = gogobj.split("\n")
    return(gogobj[:-1]) #ignore last element as this has nothing

#make the stringifyd position traces into clean numerical values
def listpostions(newsamplemarks, landmark_count, include_visibility=False):
    """Extract Tasks landmarks in CSV order, padding missing detections."""
    fields = ('x', 'y', 'z', 'visibility') if include_visibility else ('x', 'y', 'z')
    landmarks = newsamplemarks or []
    tracking_p = []
    for index in range(landmark_count):
        for field in fields:
            value = getattr(landmarks[index], field, None) if index < len(landmarks) else None
            tracking_p.append(float('nan') if value is None else value)
    return tracking_p

#We will now loop over all the videos that are present in the video file
for vidf in vfiles:
    print("We will now process video:")
    print(vidf)
    print("This is video number" + str(vfiles.index(vidf))+ "of" + str(len(vfiles)) + "videos in total")
    #capture the video, and check video settings
    videoname = vidf
    videoloc = "./Input_Videos/" + videoname
    capture = cv2.VideoCapture(videoloc) #load in the videocapture
    frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH) #check frame width
    frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) #check frame height
    samplerate = capture.get(cv2.CAP_PROP_FPS)   #fps = frames per second

    #make an 'empty' video file where we project the pose tracking on
    fourcc = cv2.VideoWriter_fourcc(*'MP4V') #for different video formats you could use e.g., *'XVID'
    out = cv2.VideoWriter(outputf_mask+videoname, fourcc, 
                          fps = samplerate, frameSize = (int(frameWidth), int(frameHeight)))

    # Run MediaPipe frame by frame using Holistic with `enable_segmentation=True` to get pose segmentation.
    time = 0
    tsbody = [markerxyzbody]   #these will be your time series objects, which start with collumn names initialized above
    tshands = [markerxyzhands] #these will be your time series objects, which start with collumn names initialized above
    tsface = [markerxyzface]   #these will be your time series objects, which start with collumn names initialized above
    # IMAGE mode preserves the previous static_image_mode=True behavior.
    options = vision.HolisticLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_mask=True,
    )
    with vision.HolisticLandmarker.create_from_options(options) as holistic:
        while (True):
            ret, image = capture.read() #read frame
            if ret == True: #if there is a frame
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #make sure the image is in RGB format
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image))
                results = holistic.detect(mp_image) #apply Mediapipe holistic processing
                # Keep the writer and drawing input as a three-channel BGR frame.
                original_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.segmentation_mask is not None:
                    segmentation_mask = results.segmentation_mask.numpy_view()
                    # Tasks can return (height, width, 1); select pixels across all BGR channels.
                    if segmentation_mask.ndim == 3 and segmentation_mask.shape[2] == 1:
                        segmentation_mask = segmentation_mask[..., 0]
                    segm_2class = 0.2 + 0.8 * segmentation_mask
                    original_image[segm_2class * (1 - segm_2class) == 0] = 0

                mp_drawing.draw_landmarks(
                    original_image, results.left_hand_landmarks, hand_connections)
                mp_drawing.draw_landmarks(
                    original_image, results.right_hand_landmarks, hand_connections)
                mp_drawing.draw_landmarks(
                    original_image, results.face_landmarks, face_connections,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(128, 128, 128), thickness=1, circle_radius=1))
                mp_drawing.draw_landmarks(
                    original_image, results.pose_landmarks, pose_connections)

                # Save one fixed-width row per frame, including missing detections.
                samplebody = listpostions(results.pose_landmarks, 33, include_visibility=True)
                samplehands = (listpostions(results.left_hand_landmarks, 21)
                               + listpostions(results.right_hand_landmarks, 21))
                sampleface = listpostions(results.face_landmarks, 478)
                tsbody.append([time] + samplebody)
                tshands.append([time] + samplehands)
                tsface.append([time] + sampleface)
                #show the video as we process (you can comment this out, if you want to run this process in the background)
                cv2.imshow("resizedimage", original_image)
                out.write(original_image) #save the frame to the new masked video
                time = time+(1000/samplerate)#update the time variable  for the next frame
            if cv2.waitKey(1) == 27: #allow the use of ESCAPE to break the loop
                   break
            if ret == False: #if there are no more frames, break the loop
                break

    #once done de-initialize all processes
    out.release()
    capture.release()
    cv2.destroyAllWindows()
     ####################################################### data to be written row-wise in csv fil
    # opening the csv file in 'w+' mode
    filebody = open(outtputf_ts + vidf[:-4]+'_body.csv', 'w+', newline ='')
    #write it
    with filebody:    
        write = csv.writer(filebody)
        write.writerows(tsbody)
     # opening the csv file in 'w+' mode
    filehands = open(outtputf_ts + vidf[:-4]+'_hands.csv', 'w+', newline ='')
    #write it
    with filehands:
        write = csv.writer(filehands)
        write.writerows(tshands)
    # opening the csv file in 'w+' mode
    fileface = open(outtputf_ts + vidf[:-4]+'_face.csv', 'w+', newline ='')
    #write it
    with fileface:    
        write = csv.writer(fileface)
        write.writerows(tsface)

print("Done with processing all folders; go look in your output folders!")  