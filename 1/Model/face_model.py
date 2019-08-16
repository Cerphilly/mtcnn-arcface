#get models for 512-D feture embeddings

import os, sys
import numpy as np
import mxnet as mx
import cv2
import sklearn
from sklearn.preprocessing import normalize
from Model.mtcnn_detector import MtcnnDetector
import time

sys.path.append("..")
from src import face_preprocess

def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx,:,:] = np.fliplr(data[idx,:,:])

    return data


def get_model(ctx, image_size, model_str, layer):

    prefix = model_str + '/model'
    epoch = int(0)

    print('loading', prefix, epoch)

    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()

    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)

    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)

    return model

class FaceModel(object):
    def __init__(self, args, ctx):
        self.args = args

        self.ctx = ctx

        _vec = args.image_size.split(',')
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))

        self.model = None
        self.ga_model = None
        if len(args.model) > 0:
            self.model = get_model(self.ctx, image_size, args.model, 'fc1')
        if len(args.ga_model) > 0:
            self.ga_model = get_model(self.ctx, image_size, args.ga_model, 'fc1')

        self.threshold = args.threshold
        self.det_minsize = 50
        self.det_threshold = [0.6, 0.7, 0.8]
        # self.det_factor = 0.9

        self.image_size = image_size

        mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')

        if args.det == 0:
            detector = MtcnnDetector(model_folder=mtcnn_path, ctx=self.ctx, num_worker=1, accurate_landmark=True, threshold=self.det_threshold)
        else:
            detector = MtcnnDetector(model_folder=mtcnn_path, ctx=self.ctx, num_worker=1, accurate_landmark=True, threshold=[0.0, 0.0, 0.2])
        self.detector = detector

    def get_input(self, face_img):#detect a first face from image and resize it
        t1 = time.time()
        ret = self.detector.detect_face(face_img, det_type=self.args.det)
        t2 = time.time()
        print("took {} seconds to detect face".format(t2 - t1))

        if ret is None:
            print("No image found")
            return None
        bbox, points = ret
        print("bbox: ", bbox, np.shape(bbox[:,0])[0])
        print("points: ", points, np.shape(points[:,0]))
        if bbox.shape[0] == 0:
            return None
        bbox = bbox[0, 0:4]
        points = points[0, :].reshape((2, 5)).T
        nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')#args.image_size
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))
        return aligned

    def get_input_new(self, face_img):#can detect multiple faces from the image
        #made this because get_input is not programmed to return boxes, and points. also it returns only 1 person's aligned face image.
        aligned = []
        t1 = time.time()
        ret = self.detector.detect_face(face_img, det_type=self.args.det)
        t2 = time.time()
        print("took {} seconds to detect face".format(t2 - t1))

        if ret is None:
            print("No faces found")
            return None, None, None

        bbox, points = ret
        if bbox.shape[0] == 0:
            print("[face_model]couldn't detect face from the image")
            return None

        for i in range(np.shape(bbox[:,0])[0]):#number of boxes
            cur_bbox = bbox[i, 0:4]
            cur_point = points[i, :].reshape((2,5)).T
            cur_nimg = face_preprocess.preprocess(face_img, cur_bbox, cur_point, image_size='112,112')
            cur_nimg = cv2.cvtColor(cur_nimg, cv2.COLOR_BGR2RGB)
            cur_aligned = np.transpose(cur_nimg, (2, 0 ,1))
            aligned.append(cur_aligned)

        return bbox, points, aligned


    def get_feature(self, aligned):#get a feature from a face

        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = sklearn.preprocessing.normalize(embedding).flatten()
        return embedding


    def get_feature_new(self, aligned):#can get multiple features from the images
        embedding = []
        for i in range(len(aligned)):
            input_blob = np.expand_dims(aligned[i], axis=0)
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,))
            self.model.forward(db, is_train=False)
            cur_embedding = self.model.get_outputs()[0].asnumpy()
            cur_embedding = sklearn.preprocessing.normalize(cur_embedding).flatten()

            embedding.append(cur_embedding)

        return embedding


    def get_ga(self, aligned):#get gender and age from an aligned image.
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.ga_model.forward(db, is_train=False)
        ret = self.ga_model.get_outputs()[0].asnumpy()
        g = ret[:, 0:2].flatten()
        gender = np.argmax(g)
        a = ret[:, 2:202].reshape((100, 2))
        a = np.argmax(a, axis=1)
        age = int(sum(a))

        return gender, age



    def get_image(self, img, points):#get image chips of face
        '''chips = detector.extract_image_chips(img, points, 144, 0.37)'''
        new_face = self.detector.extract_image_chips(img, points)

        return new_face








