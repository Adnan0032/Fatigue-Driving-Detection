# Import necessary libraries
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

# Function to produce an audible alert using the "espeak" command
def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    # Run the alarm in a loop while alarm_status is True
    while alarm_status:
        print('call')
        # Execute the "espeak" command to speak the message
        s = 'espeak "' + msg + '"'
        os.system(s)

    # If alarm_status2 is True, say the message and set saying to False
    if alarm_status2:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2 * C)

    return ear

# Function to calculate the average EAR for both eyes
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

# Function to calculate the vertical distance between the upper and lower lips
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

# Constants for eye aspect ratio (EAR) and consecutive frames for drowsiness alert
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 40

# Threshold for yawn detection
YAWN_THRESH = 20

# Global variables for alarm and status tracking
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

# Print loading messages
print("-> Loading the predictor and detector...")
# Use Haarcascade classifier for face detection (faster but less accurate)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# Load the shape predictor for facial landmarks
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Print start message
print("-> Starting Video Stream")

# Start the video stream with the specified webcam index
vs = VideoStream(src=args["webcam"]).start()
# For Raspberry Pi camera, use the following line
# vs = VideoStream(usePiCamera=True).start()

# Allow the camera to warm up
time.sleep(1.0)

# Main loop
while True:
    # Read a frame from the video stream and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haarcascade classifier
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Loop over the detected faces
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        # Get facial landmarks using the shape predictor
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Calculate eye aspect ratio (EAR) and eye contours
        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        # Calculate lip distance for yawn detection
        distance = lip_distance(shape)

        # Draw contours for eyes and lips on the frame
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        # Check for drowsiness based on eye aspect ratio
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # If consecutive frames indicate drowsiness, trigger an alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if alarm_status == False:
                    alarm_status = True
                    # Start a thread for the alarm function
                    t = Thread(target=alarm, args=('wake up sir',))
                    t.daemon = True
                    t.start()

                # Display drowsiness alert on the frame
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            alarm_status = False

        # Check for yawn based on lip distance
        if distance > YAWN_THRESH:
            # If yawn is detected, trigger an alarm
            cv2.putText(frame, "Yawn Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if alarm_status2 == False and saying == False:
                alarm_status2 = True
                # Start a thread for the alarm function
                t = Thread(target=alarm, args=('take some fresh air sir',))
                t.daemon = True
                t.start()
        else:
            alarm_status2 = False

        # Display eye aspect ratio and lip distance on the frame
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Check if the 'q' key is pressed to exit the loop
    if key == ord("q"):
        break

# Cleanup: Close all windows and stop the video stream
cv2.destroyAllWindows()
vs.stop()
