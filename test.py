# from tensorflow.keras.models import load_model
# from time import sleep
# from tensorflow.keras.preprocessing.image import img_to_array
# from keras.preprocessing import image
# import cv2
# import numpy as np
# from keras import models
# import keras

# face_classifier = cv2.CascadeClassifier(r'C:\Users\Alok\Downloads\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
# classifier =load_model(r'C:\Users\Alok\Downloads\Emotion_Detection_CNN-main\model.h5')

# emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# cap = cv2.VideoCapture(0)

# while True:
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray)
#     canvas = np.zeros((250, 300, 3), dtype="uint8")
#     frameClone = frame.copy()
#     # for (x,y,w,h) in faces:
#     if len(faces) > 0:
#         faces = sorted(faces, reverse=True,
#         key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
#         (x, y, w, h) = faces
#         #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
#         roi_gray = gray[y:y+h,x:x+w]
#         roi_gray = cv2.resize(roi_gray,(56,56))

    
#         if np.sum([roi_gray])!=0:
#             roi = roi_gray.astype('float')/255.0
#             roi = img_to_array(roi)
#             roi = np.expand_dims(roi,axis=0)
#             prediction = classifier.predict(roi)[0]
#             emotion_probability = np.max(prediction)
#             label=emotion_labels[prediction.argmax()]
#             # label_position = (x,y)
            
#         for (i, (emotion, prob)) in enumerate(zip(emotion_labels, prediction)):
#             text = "{}: {:.2f}%".format(emotion, prob * 100)
#             w = int(prob * 300)
#             cv2.rectangle(canvas, (7, (i * 35) + 5),
#             (w, (i * 35) + 35), (0, 0, 255), -1)
#             cv2.putText(canvas, text, (10, (i * 35) + 23),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.45,
#             (255, 255, 255), 2)
#             cv2.putText(frame, label, (x, y - 10),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
#             cv2.rectangle(frame, (x, y), (x + w, y + h),
#                               (0, 255, 0), 2)

#     cv2.imshow('your_face', frame)
#     cv2.imshow("Probabilities", canvas)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
# #             cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
# #         else:
# #             cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
# #     cv2.imshow('Emotion Detector',frame)
# #     cv2.imshow("Probabilities", canvas)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()


from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from keras import models
import keras

face_classifier = cv2.CascadeClassifier(r'C:\Users\Alok\Downloads\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\Alok\Downloads\Emotion_Detection_CNN-main\xception.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

while True:
    frame = cap.read()[1]
    #frame = imutils.resize(frame,width=400)
    #frame = cv2.resize(frame, dsize= (480, 400))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    # for (x,y,w,h) in faces:
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
        roi_gray = gray[fY:fY + fH, fX:fX + fW]
        roi_gray = cv2.resize(roi_gray,(56,56))

    
        
        roi = roi_gray.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi,axis=0)
        prediction = classifier.predict(roi)[0]
        emotion_probability = np.max(prediction)
        label=emotion_labels[prediction.argmax()]
            # label_position = (x,y)
            
        for (i, (emotion, prob)) in enumerate(zip(emotion_labels, prediction)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5),
            (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 2)
            cv2.putText(frame, label, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)

    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
#             cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#         else:
#             cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#     cv2.imshow('Emotion Detector',frame)
#     cv2.imshow("Probabilities", canvas)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()