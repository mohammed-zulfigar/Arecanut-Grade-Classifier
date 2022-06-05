import tensorflow.keras
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils


np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('arecanut_ grade_classifier.h5')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# initialize the video stream
print("[INFO] starting video stream...")
cam = VideoStream(src=0).start()
# cam = cv2.VideoCapture(r"dataset\test\badVid.mp4")

text = ""

while True:
    img = cam.read()
    img = cv2.resize(img,(224, 224))

    # img = imutils.resize(img, width=224)
    # img = imutils.resize(img, height=224)

    # comment(16,21,22) and uncomment below to input image 
    # img = cv2.imread('dataset/test/b.jpg')
    # img = cv2.resize(img,(224, 224))

    image_array = np.asarray(img)

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    
    for i in prediction:
        if i[0] > 0.7:
            text ="Good"
        if i[1] > 0.7:
            text ="Bad"
        # print(text)
        img = cv2.resize(img,(500, 500))
        cv2.putText(img,text,(10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),1)
    cv2.imshow('img',img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break