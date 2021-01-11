# import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import cv2

dataDirectory = "Directory" ## training dataset
classes = ["WithMask","WithOutMask"] #list of classes

# extracting the data
# changing the image size() to 224*224 #
img_size = 224
training_data = []
# making a function to extract data #
def create_trainin_data():
    for category in classes:
        path = os.path.join(dataDirectory,category)
        class_num = classes.index(category)     ## 0 1 , ##
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(img_size,img_size))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass


create_trainin_data()
# suffling randomly
random.shuffle(training_data)

X = []
y = []

for feature,label in training_data:
    X.append(feature)
    y.append(label)

X = np.array(X).reshape(-1,img_size,img_size,3)
Y = np.array(y)
# normalize the data
x = X/255.0;

# generalising data
import pickle
# for X:
pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()
# ................................
# for Y:
pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()


# ........................importing Deep Learning Model ................#
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = tf.keras.applications.mobilenet.MobileNet() # model #

base_input = model.layers[0].input
base_output = model.layers[-4].output

Flat_layer = layers.Flatten()(base_output)
final_output = layers.Dense(1)(Flat_layer) ## 0 ,1 ##
final_output = layers.Activation('sigmoid')(final_output)

new_model = keras.Model(inputs = base_input,outputs =final_output)

new_model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
new_model.fit(X,Y,epochs=1,validation_split=0.1)

# checking the network the predictions for real faces
frame = cv2.imread("12.jpg")

facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# Detection of faces in reactange from x->x+w and y->y+h
faces  = facecascade.detectMultiScale(gray,1.1,4)
for x,y,w,h in faces:
    roi_gray = gray[y:y+h , x:x+w]
    roi_color = frame[y:y+h , x:x+w]
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    faces = facecascade.detectMultiScale(roi_gray)
    if len(faces) == 0:
        print("face not Detected")
    else:
        for (ex,ey,ew,eh) in faces:
            face_roi = roi_color[ey:ey:eh,ex:ex+ew]



 # checking the image;
final_image = cv2.resize(face_roi,(224,224))
final_image = np.expand_dims(final_image,axis=0)
final_image = final_image/255.0

Predictions = new_model.predict(final_image)

print(Predictions)

# cool part -> checking with Laptop cam
import cv2
path  = "haarcascade_frontalface_default.xml"
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

# set the rectangle to default white
recatangle_bgr = (255,255,255)
# make a black image
img = np.zeros((500,500))

text = "Some text in a box"
(text_width,text_height) = cv2.getTextSize(text,font,fontScale=font_scale,thickness=1)[0]
# set  the start postion
text_offset_x = 10
text_offset_y = img.shape[0] - 25
# box coordinate
box_cords = ((text_offset_x,text_offset_y),(text_offset_x+text_width+2,text_offset_y-text_height-2))
cv2.rectangle(img,box_cords[0],box_cords[1],recatangle_bgr,cv2.FILLED)
cv2.putText(img,text,(text_offset_x,text_offset_y),font,fontScale=font_scale,color=(0,0,0),thickness=1)

cap = cv2.VideoCapture(1)

# check if camera is on or not
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open webcam")

while True:
    ret,frame = cap.read()
    faceCasede  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = facecascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y + h, x:x + w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        if len(faces):
            print("face not detected")
        else:
            for (ex,ey,ew,eh) in faces:
                face_roi = roi_color[ey:ey+eh,ex:ex+ew]

    final_image = cv2.resize(face_roi,(244,244))
    final_image = np.expand_dims(final_image,axis = 0 ) #   need for fouth dimension
    font = cv2.FONT_HERSHEY_PLAIN
    Predictions = new_model.predict(final_image)

    font_scale = 1.5
    font = cv2.FONT_HERSHEY_COMPLEX

    # now final work if prediction > 0 indicate there is no mask else there is mask
    if (Predictions>0):
        status = "NO Mask"

        x1,y1,w1,h1 = 0,0,175,75

        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)

        cv2.putText(frame,status,(x1 + int(x1/10),y1 + int(y1/10)),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)

        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))

    else:

        x1,y1,w1,h1 = 0,0,175,75

        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)

        cv2.putText(frame,status,(x1 + int(x1/10),y1 + int(y1/10)),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)

        cv2.putText(frame,status,(100,150),font,3,(0,255,0),2,cv2.LINE_4)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0))


    cv2.imshow('Face Mask Detection Tutorial ',frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



