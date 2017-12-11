#!/usr/bin/env python
import sys
import dlib
import cv2
from skimage import io


cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:
        break

cv2.destroyWindow("preview")

# Gets file name from command line
file_name = sys.argv[1]

# Pre-trained face detection model
predictor_model = "shape_predictor_68_face_landmarks.dat"

# Creates HOG face detector and model for face landmarks
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)

win = dlib.image_window()

# Load the image
image = io.imread(file_name)

# Run the HOG detector on image data
detected_faces = face_detector(image, 1)

print("I found {} faces in the file {}".format(len(detected_faces), file_name))

# Show the desktop window with the image
win.set_image(image)

# Loop through each face that was found
for i, face_rect in enumerate(detected_faces):
    print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

    # Draw a box around each face
    win.add_overlay(face_rect)

    # Get the face's pose
    pose_landmarks = face_pose_predictor(image, face_rect)

    # Draw landmarks on the face
    win.add_overlay(pose_landmarks)

dlib.hit_enter_to_continue()