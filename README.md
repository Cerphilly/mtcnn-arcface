# mtcnn-arcface
mtcnn detection + arc face

Program made by Yoonhee Gil in 2019/07~ 2019/08

Made from insightface of deepinsight (https://github.com/deepinsight/insightface)

In Application folder, demo.py(or demo_rasp.py) gets webcam image and detects faces in the camera.
Then it compares every faces with faces in Database(Images deleted for privacy) using 512-D features of each faces.
If there's new face in the camera, then it saves its image and 512-D features in the database. If the face is already in the database, then it prints the most similar face in the database.

You can test a image file in test.py.

MTCNN Detector uses pretrained model in Model/mtcnn-model, and Arcface used resnet100(model-r100-ii) for face recognition. You can see other face recognition models in Pretrained_model/__init__.py.
You can change face recognition models by changing parser. All the Pretrained models in this program is also from Insightface.

Since Insightface's mtcnn detector is somewhat programmed to get only one face from the image file input. Thus I added new function in Model/face_model.py. 
Use get_input_new and get_feature_new to get multiple faces and features from the image.

Also, MTCNN can get facial landmarks(eyes, nose, mouth) also, but I deactivated it being drawn in output image file.
Deactivate commentation in draw function in src/tools.py to see facial landmarks.

You can also get gender and age from each faces. Add gamodel-r50 in the ga-model in parser. However, it is not that precise, even to Asian faces(It is said that this model is trained by AsianFace.)


