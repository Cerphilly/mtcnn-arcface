# from camera, detect faces and extract 512-D features. See this people are in the database in realtime.
# can save faces images and 512-D features
import cv2
import numpy as np
import os, sys
import argparse
import mxnet as mx
import datetime, time
import requests
import json

sys.path.append('..')
from Camera import camera
from Model import face_model
from Database import face_db
from src import tools

parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/Users/aham/PycharmProjects/1/Pretrained_model/model-r100-ii',
                    help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load gender age model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from beginning')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--score', default=0.99, type=float,
                    help='if detected face''s score is lower than this, than it will not be detected as face')  # 0.999여도 좋을지도
parser.add_argument('--similarity', default=0.55, type=float,
                    help='if similarity between two face is above this means those two are the same face')
parser.add_argument('--save', default=True, type=bool,
                    help='True if you want to save images and npy of faces in this demo')
# what is the best similarity? 0.55, maybe.
args = parser.parse_args()


EndPoint = "http://192.168.0.12:5000/face_recog"

class Demo():

    def __init__(self):

        self.model = face_model.FaceModel(args, ctx=mx.cpu())
        self.db = face_db.Database()
        self.camera = camera.Camera(url='http://aham_demo_raspi1.com.ngrok.io/?action=snapshot')
        self.camera.picam_run()
        #self.camera = camera.Camera(camera_address=0)
        #self.camera.run()
        self.time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        print("current time is: ", self.time)

    def process(self):
        while True:
            self.time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

            t1 = time.time()

            self.frame = self.camera.picam_run()
            #self.ret, self.frame = self.camera.cam.read()

            self.frame = cv2.resize(self.frame, (640, 360))

            boxes, points, results = self.model.get_input_new(face_img=self.frame)
            if boxes is not None:
                print(np.shape(boxes[:, 0])[0], "people found")

                for i in range(np.shape(boxes[:, 0])[0]):
                    print(f'{i}: score {boxes[i, :][4]}')

            boxes, points, results = tools.detector(boxes, points, results, args.score)

            tools.draw(boxes, points, self.frame)
            self.recognize(points, results)

            t2 = time.time()
            print("took {} seconds\n".format(t2 - t1))

            cv2.imshow("detection result", self.frame)
            cv2.waitKey(30)


    def recognize(self, points, results, save=args.save):  # see this person is in the database. save: save new faces in the database
        if results is None:
            return

        list = os.listdir(self.db.npy_path())
        features = self.model.get_feature_new(results)
        new_faces = self.model.get_image(img=self.frame, points=points)
        cam_face = 0

        if (len(list)) == 0:
            print("No faces saved in the database")
            if (save == True):

                self.db.save(time=self.time, img=new_faces[0], data=features[0])
                print("saved first feature and image in the list")

            elif (save == False):
                print("no data saved in the database. returning..")
                return

        for f in features:
            new_person = True  # db에 모든 사람들에게 대해 true여야 함
            high_sim = 0  # 여러개의 파일이 similarity 0.5를 넘으면 그 중 가장 높은 similarity가 그 사람 얼굴인 것으로
            sim_num = 0 #db내 비슷한 사람들 수
            for l in list:
                sim = self.db.sim(f, self.db.npy_load(l))
                if sim > args.similarity:
                    sim_num += 1
                    new_person = False
                    if sim > high_sim:
                        high_sim = sim
                        similar_face = l

                print("{}: similarity btw {} is {}".format(cam_face, l, sim))


            if new_person == True:
                print("{}: found new person\n".format(cam_face))
                if save == True:

                    self.db.save(time=self.time, img=new_faces[cam_face], data=f)
                    print("{}: new data saved as {}".format(cam_face, self.time))
                    data = self.db.json_load(name=self.time)
                    print(data)
                    requests.post(url=EndPoint, data=json.dumps(data))
                    print("{}: sent json file {}.json".format(cam_face, self.time))



            elif new_person == False:
                print("{}: looks like {}.jpg\n".format(cam_face, os.path.splitext(similar_face)[0]))
                data = self.db.json_load(name=os.path.splitext(similar_face)[0])
                print(data)
                requests.post(url=EndPoint, data=json.dumps(data))
                print("{}: sent json file {}.json".format(cam_face, os.path.splitext(similar_face)[0]))

            cam_face += 1
            print("sim_num: {}".format(sim_num))




if __name__ == '__main__':
    test = Demo()
    test.process()