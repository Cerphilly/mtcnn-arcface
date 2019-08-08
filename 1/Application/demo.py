#from camera, detect faces and extract 512-D features. See this people are in the database in realtime.
#can save faces images and 512-D features
import cv2
import numpy as np
import os, sys
import argparse
import mxnet as mx
import datetime, time

sys.path.append('..')
from Camera import camera
from Model import face_model
from Database import face_db
from src import tools


parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/Users/aham/PycharmProjects/1/Pretrained_model/model-r100-ii', help='path to load model.')
#512-D feature is dependent to model. it means same face feature from different model will not be similar at all.
parser.add_argument('--ga-model', default='', help='path to load gender age model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from beginning')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--score', default=0.99, type=float, help='if detected face''s score is lower than this, than it will not be detected as face')#0.999여도 좋을지도
parser.add_argument('--similarity', default=0.55, type=float, help='if similarity between two face is above this means those two are the same face')
parser.add_argument('--save', default=True, type=bool, help='True if you want to save images and npy of faces in this demo')
#what is the best similarity?
args = parser.parse_args()


class Demo():

    def __init__(self):

        self.model = face_model.FaceModel(args, ctx=mx.cpu())#face model initialization. when to use gpu, set ctx = mx.gpu(args.gpu)
        self.db = face_db.Database()#Database initialization
        self.camera = camera.Camera(camera_address=0)#Webcam initialization
        self.camera.run()#start camera
        self.time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        print("current time is: " , self.time)#print demo start time


    def process(self):
        while True:
            self.time = datetime.datetime.now()
            self.time = self.time.strftime('%Y%m%d-%H%M%S')#get current time at every loop
            t1 = time.time()

            self.ret, self.frame = self.camera.cam.read()#self.ret is true when the video is in its end. But it is not used in this code.
            self.frame = cv2.resize(self.frame, (640,360))#resize image. If not resized, it'll take too much time to process each frame.

            boxes, points, results = self.model.get_input_new(face_img=self.frame)#get face boxes, points(eye, nose, mouth), aligned images.
            if boxes is not None:#boxes's shape: (n, 5) , n: number of people in a frame
                print(np.shape(boxes[:, 0])[0], "people found")#n people found

                for i in range(np.shape(boxes[:, 0])[0]):
                    print(f'{i}: score {boxes[i][4]}')#boxes[i][4]: score(or confidence) of each face. >0.999 for perfect face image


            boxes, points, results = tools.detector(boxes, points, results, args.score)

            tools.draw(boxes, points, self.frame)#draw boxes and points in the frame
            self.recognize(points, results)#see whether this face is in db or not

            t2 = time.time()
            print("took {} seconds\n".format(t2-t1))#print the time took for one loop. if ctx = mx.cpu(): about 0.30s


            cv2.imshow("detection result", self.frame)
            cv2.waitKey(30)


    def recognize(self, points, results, save=args.save):#see this person is in the database. save: save new faces in the database
        if results is None:
            return
        list = os.listdir(self.db.npy_path())#list of npy files in Database/npy folder
        features = self.model.get_feature_new(results)#512-D features of faces
        new_faces = self.model.get_image(img=self.frame, points=points)#image file of faces
        cam_face = 0#how many people in the camera?

        if(len(list)) == 0:#if no one is in the db, save the 1st person in db
            print("No faces saved in the database")
            if(save==True):
                self.db.save(time=self.time, img=new_faces[0], data=features[0])
                print("saved first feature and image in the list")

            elif(save==False):
                print("no data saved in the database. returning..")
                return

        for f in features:
            new_person=True#has to be true to all people in the db
            high_sim = 0#if many features is hihger than args.similarity, then a feature with highest similarity is considered as same face.

            for l in list:
                sim = self.db.sim(f, self.db.npy_load(l))#check similarity btw all the features in db.
                if sim > args.similarity:
                    new_person = False
                    if sim > high_sim:
                        high_sim = sim
                        simliar_face = l

                print("{}: similarity btw {} is {}".format(cam_face, l, sim))

            if new_person==True:#if the face's similarity btw all the files in db is below args.similarity, then it is the new face.
                print("{}: found new person\n".format(cam_face))
                if save == True:
                    self.db.save(time=self.time, img=new_faces[cam_face], data=f)#save the current time, face image, and feature in db
                    print("{}: new data saved as {}".format(cam_face, self.time))


            elif new_person==False:
                print("{}: looks like {}.jpg\n".format(cam_face, os.path.splitext(simliar_face)[0]))

            cam_face += 1


if __name__ == '__main__':
    test = Demo()
    test.process()