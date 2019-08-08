import cv2
import requests
import numpy as np

'''
This server:
    Input: Camera address
    Process: Start a camera thread
    Output: Run an overridden process function for each frame
'''

class Camera(object):

    in_progress = False

    def __init__(self, camera_address=None, url=None):
        self.camera_address = camera_address
        self.url = url


    def get_status(self):
        return self.in_progress

    def run(self):
        print('[Camera] Camera is initializing ...')
        if self.camera_address is not None:
            self.cam = cv2.VideoCapture(self.camera_address)
            self.in_progress = True
        else:
            print('[Camera] Camera is not available!')
            return


    def picam_run(self):
        print('[Camera] Pi cam is initializing ...')
        if self.url is not None:
            image = requests.get(url=self.url)
            img = cv2.imdecode(np.asarray(bytearray(image.content)), 1)

            self.cam = img
            self.in_progress = True
            return self.cam

        else:
            print('[Camera] Pi Camera is not available!')
            return



