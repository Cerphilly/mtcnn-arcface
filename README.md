# mtcnn-arcface
mtcnn detection + arc face

Program made by Yoonhee Gil in 2019/07~ 2019/08

Made from insightface of deepinsight (https://github.com/deepinsight/insightface)

In __Application__ folder, demo.py(or demo_rasp.py) gets webcam image and detects faces in the camera.
Then it compares every faces with faces in Database(Images deleted for privacy) using 512-D features of each faces.
If there's new face in the camera, then it saves its image and 512-D features in the database. If the face is already in the database, then it prints the most similar face in the database.

You can test any image file you want in test.py.

MTCNN Detector uses pretrained model in Model/mtcnn-model, and Arcface used resnet100(model-r100-ii) for face recognition. You can see other face recognition models in Pretrained_model/init.py.
You can change face recognition models by changing parser. All the Pretrained models in this program is also from Insightface.

There was a img folder in Database folder, but I deleted it due to privacy issue. Make your own to use database.

__What I have changed from Insightface__

Though all features from this program is from insightface, there are few things that have changed.

1. Don't know why, but Insightface's mtcnn detector is somewhat programmed to get only one face from the image file input. Thus I added new function in Model/face_model.py. 
Use _get_input_new_ and _get_feature_new_ to get multiple faces and features from the image.

2. Also, MTCNN can get facial landmarks(eyes, nose, mouth) also, but I deactivated it being drawn in output image file.
Deactivate commentation in draw function in src/tools.py to see facial landmarks.

3. MTCNN detector is a powerful detector, but can detect things that are not faces. So I added detector function in src/tools. It gets confidence of each faces and return only those who have higher confidence than your criteria. You can change the criteria in parser of demo or test. 0.99 is the default criteria. 

4. You can also get __gender and age__ from each faces. Add gamodel-r50 in the ga-model in parser. However, it is not that precise, even to Asian faces(It is said that this model is trained by AsianFace.)

5. You can deactivate your face information and image being saved in database. Change __save__ in parser.
