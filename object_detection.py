import os
import cv2
import numpy as np
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter

import matplotlib
import matplotlib.pyplot as plt

### Define function for inferencing with TFLite model and displaying results

def tflite_detect_images(modelpath, image, lblpath, min_conf=0.5, num_test_images=10, txt_only=False, name_error='loi'):

    # Load the label map into memory
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence of detected objects

    detections = []

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    object_detected = False  # Flag to check if any object is detected
    for i in range(len(scores)):
        if (scores[i] > min_conf) and (scores[i] <= 1.0):
            object_detected = True
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
            label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10),
                          (255, 255, 255), cv2.FILLED)  # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Draw label text

            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
            # print(scores[i]*100)
    if object_detected:
        detected_objects = ', '.join([det[0] for det in detections])
        # error_sp = name_error + ', ' + detected_objects
        error_sp = name_error 
    else:
        error_sp = 0

    return image, detections, error_sp


# cap = cv2.VideoCapture(0)

# # Set up variables for running user's model
# PATH_TO_MODEL = 'D:/study/NCKH/opencv/NCKH/step/detect.tflite'  # Path to .tflite model file
# PATH_TO_LABELS = 'D:/study/NCKH/opencv/NCKH/step/labelmap.txt'  # Path to labelmap.txt file
# min_conf_threshold = 0.8  # Confidence threshold (try changing this to 0.01 if you don't see any detection results)
# images_to_test = 10  # Number of images to run detection on

# while True:
#     # ret, frame = cap.read()

#     frame = cv2.imread('D:/study/NCKH/tensorflow/img_train/final/images/0.jpg')
#     # Run inferencing function!
#     result_frame, detections, error_sp = tflite_detect_images(PATH_TO_MODEL, frame, PATH_TO_LABELS, min_conf_threshold, images_to_test, name_error='Object detected')

#     # if detections:
#     #     detected_objects = ', '.join([det[0] for det in detections])
#     #     print("Detected objects:", detected_objects)
#     # else:
#     print(error_sp)

#     # Show the resulting frame
#     cv2.imshow('Object Detection', result_frame)

#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break

# # Release the camera
# # cap.release()
# # cv2.destroyAllWindows()
