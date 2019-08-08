#tools
import cv2
import numpy as np
import os

def draw(boxes=None, points=None, image=None):  # draw boxes and points in the frame

    if boxes is not None:
        for b in boxes:
            cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color=(255, 255, 255),
                          thickness=2)  # RGB color: (RED, GREEN, BLUE)
    '''
    if points is not None:
        for p in points:
            for i in range(5):
                cv2.circle(image, (p[i], p[i + 5]), 1, color=(255, 0, 0), thickness=2)
    '''

    return image


def get_abs_path_list(dir_path, ext):
    list = os.listdir(dir_path)
    for l in list:
        if os.path.splitext(l)[-1] != ext:
            list.remove(l)
    list = sorted(list)

    return list

def detector(boxes, points, results, score):#to detect faces with score over threshold
    new_boxes=[]
    new_points=[]
    new_results=[]

    if boxes is not None:

        for i in range(np.shape(boxes[:, 0])[0]):
            if (boxes[i][4] > score):
                new_boxes.append(boxes[i])
                new_points.append(points[i])
                new_results.append(results[i])

    return new_boxes, new_points, new_results


def size(boxes):#return the sizes of multiple faces in a image
    sizes = []
    for i in range(np.shape(boxes[:,0])[0]):
        size = np.sqrt(np.square(boxes[i,:][0] - boxes[i,:][2]) + np.square(boxes[i,:][1] - boxes[i,:][3]))
        sizes.append(size)

    return sizes


def dist(f1, f2):#distance btw 2 features
    return np.sum(np.square(f1-f2))

def sim(f1, f2):#similarity btw 2 features
    return np.dot(f1, f2.T)

