#get 512D hyperspace feature from wanted images

import mxnet as mx

import cv2
import os, sys
import argparse
import numpy as np

from Model import face_model
from Database import face_db
sys.path.append("..")
from src import tools
import time

parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/Users/aham/PycharmProjects/1/Pretrained_model/model-r100-ii', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load gender age model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--score', default=0.99, type=float, help='if detected face''s score is lower than this, than it will not be detected as face')#0.999여도 좋을지도
parser.add_argument('--similarity', default=0.55, type=float, help='if similarity between two face is above this means those two are the same face')
parser.add_argument('--save', default=True, type=bool, help='True if you want to save images and npy of faces in this demo')

args = parser.parse_args()

IMAGE_PATH = 'oscar1.jpg'
db = face_db.Database()
model = face_model.FaceModel(args, ctx=mx.cpu())

if __name__ == '__main__':
    t1 = time.time()
    img = cv2.imread(IMAGE_PATH, 1)
    #img = cv2.resize(img, (640, 360))
    boxes, points, results = model.get_input_new(face_img=img)

    if boxes is not None:  # boxes's shape: (n, 5) , n: number of people in a frame
        print(np.shape(boxes[:, 0])[0], "people found")  # n people found

        for i in range(np.shape(boxes[:, 0])[0]):
            print(f'{i}: score {boxes[i][4]}')  # boxes[i][4]: score(or confidence) of each face. >0.999 for perfect face image

    boxes, points, results = tools.detector(boxes, points, results, args.score)
    tools.draw(boxes, points, img)

    print(np.shape(boxes[:,0])[0], "people detected")

    features = model.get_feature_new(results)

    #for i in range(len(features)):
    #    for j in range(len(features)):
    #        print("similarity btw {} and {} is {}".format(i, j, db.sim(features[i], features[j])))

    #print(features)
    t2 = time.time()
    print(f"it took {t2-t1} seconds")

    cv2.imshow(IMAGE_PATH, img)

    cv2.waitKey(0)











