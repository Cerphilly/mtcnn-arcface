#save features of image and calculate distance similarity. Can save images if you want to.
import os
import numpy as np
import cv2
import datetime
import json
#
DB_PATH = os.path.dirname(os.path.abspath(__file__))
NPY_PATH = os.path.join(DB_PATH, 'npy')#npy is the fatest
IMG_PATH = os.path.join(DB_PATH, 'img')
JSON_PATH = os.path.join(DB_PATH, 'json')

current_time = datetime.datetime.now()
current_time = current_time.strftime('%Y%m%d-%H%M%S')

class Database(object):

    def db_path(self):
        return DB_PATH

    def npy_path(self):
        return NPY_PATH

    def img_path(self):
        return IMG_PATH

    def json_path(self):
        return JSON_PATH

    def save(self, time, img, data, name=None, age=None):
        self.json_save(time=time, name=name, age=age)
        self.img_save(time=time, img=img)
        self.npy_save(time=time, data=data)


    def json_save(self, time, name=None, age=None):
        if name==None: name='Unknown'
        if age==None: age='Unknown'

        if name == 'Unknown':
            file_path = os.path.join(JSON_PATH, time)
        else:
            file_path = os.path.join(JSON_PATH, name)

        data = {
            'time': time,
            'name': name,
            'age': age,
            "img_url": "Unknown"
        }

        with open("{}.json".format(file_path), 'w') as f:
            json.dump(data, f, ensure_ascii=False)


        pass

    def img_save(self, time, img, name=None):
        if name==None:
            file_name = os.path.join(IMG_PATH, time)
        else:
            file_name = os.path.join(IMG_PATH, name)

        cv2.imwrite('{}.jpg'.format(file_name), img=img)


    def npy_save(self, time, data,  name=None):
        if name==None:
            file_name = os.path.join(NPY_PATH, time)
        else:
            file_name = os.path.join(NPY_PATH, name)

        np.save(file_name, data)


    def npy_load(self, name):
        file_name = name
        if os.path.isfile(file_name):#name -> absolute path
            return np.load(file_name)
        else:
            file_name = os.path.join(NPY_PATH, file_name)#name == 'dfdf.npy'
            if os.path.isfile(file_name):
                return np.load(file_name)
            else:
                file_name = file_name + '.npy'#name == 'dfdf'
                if os.path.isfile(file_name):
                    return np.load(file_name)
                else:#cannot find the file
                    print("[face_db]cannot find the file {}".format(name))


    def npy_load_all(self):
        npy = []
        list = []
        filenames = os.listdir(NPY_PATH)
        for (path, dir, files) in os.walk(NPY_PATH):
            for filename in filenames:
                full_filename = os.path.join(NPY_PATH, filename)
                list.append(full_filename)
        for i in range(len(list)):
            npy.append(np.load(list[i]))

        return npy

    def json_load(self, name, type='json'):
        assert type in ('json', 'str')
        file = {}
        file_name = name
        if os.path.isfile(file_name):#absolute path
            with open(file_name) as f:
                file = json.load(f)
                if(type=='str'):
                    file = str(file)
                return file
        else:
            file_name = os.path.join(JSON_PATH, file_name)#afdf.json
            if os.path.isfile(file_name):
                with open(file_name) as f:
                    file = json.load(f)
                    if(type=='str'):
                        file = str(file)
                    return file
            else:
                file_name = file_name + '.json'#afdf
                if os.path.isfile(file_name):
                    with open(file_name) as f:
                        file = json.load(f)
                        if(type=='str'):
                            file = str(file)
                        return file
                else:
                    print("[face_db]cannot find file {}".format(name))


    def dist(self, f1, f2):
        return np.sum(np.square(f1-f2))


    def sim(self, f1, f2):
        return np.dot(f1, f2.T)

