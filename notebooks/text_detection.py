import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

layers = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']
east = cv2.dnn.readNet('../youtube_data/frozen_east_text_detection.pb')

# credit: https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV/blob/master/opencv_text_detection_image.py
def find_text(image, min_confidence=0.5):
    image = pad_image(image)
    rgb_mean = (123.68, 116.78, 103.94)
    blob = cv2.dnn.blobFromImage(image, mean=rgb_mean, swapRB=True)
    east.setInput(blob)
    scores, geometry = east.forward(layers)
    return bounding_boxes(scores, geometry, min_confidence)

def pad_image(image):
    deep_w, deep_h = 128, 96
    padded = np.zeros((deep_h, deep_w, 3), dtype=np.uint8)
    h, w, _ = image.shape
    padded[0:h, 0:w, :] = image
    return padded

def bounding_boxes(scores, geometry, min_confidence):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            bbox_h = xData0[x] + xData2[x]
            bbox_w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - bbox_w)
            startY = int(endY - bbox_h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return non_max_suppression(np.array(rects), probs=confidences)

def draw_bounding_boxes(image, boxes):
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

def boxes_area(image, boxes):
    w, h, _ = image.shape
    bitmap = np.zeros((w, h), dtype=bool)
    for (startX, startY, endX, endY) in boxes:
        bitmap[startY:endY, startX:endX] = True
    return bitmap.mean()
