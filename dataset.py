# import cv2
# import numpy as np
#
# face_classifier = cv2.CascadeClassifier('C:/Users/Omkari Sneha/OneDrive/Desktop/haarcascade_frontalface_default.xml')
# def face_extractor(img):
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray,1.3,5)
#     if faces is():
#         return None
#     for(x,y,w,h) in faces:
#         cropped_face = img[y:y+h, x:x+w]
#     return cropped_face
# cap = cv2.VideoCapture(0)
# count = 0
# while True:
#     ret, frame = cap.read()
#     if face_extractor(frame) is not None:
#         count = count+1
#         face = cv2.resize(face_extractor(frame),(200,200))
#         face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#
#         file_name_path ='C:/Users/Omkari Sneha/OneDrive/Desktop/DataSet/'+str(count)+'.jpg'
#         cv2.imwrite(file_name_path,face)
#         cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
#         cv2.imshow('Face Cropper',face)
#     else:
#         print("Face Not Found")
#         pass
#     if cv2.waitKey(1) == 13 or count == 100:
#         break
#
#     cap.release()
#     cv2.destroyWindows()
#
#     print("Data set collection completed")
import cv2
import numpy as np
import os

# Load Haar Cascade Classifier
face_classifier = cv2.CascadeClassifier('C:/Users/Omkari Sneha/OneDrive/Desktop/haarcascade_frontalface_default.xml')

# Create dataset directory if it does not exist
dataset_path = 'C:/Users/Omkari Sneha/OneDrive/Desktop/DataSet/'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)


# Function to extract face from a frame
def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        cropped_face = img[y:y + h, x:x + w]
        return cropped_face
    return None


# Initialize camera
cap = cv2.VideoCapture(0)
count = 0

print("Starting dataset collection...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        continue

    face = face_extractor(frame)

    if face is not None:
        count += 1
        face = cv2.resize(face, (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save the captured face
        file_name_path = os.path.join(dataset_path, f"{count}.jpg")
        cv2.imwrite(file_name_path, face)

        # Display the face and counter
        cv2.putText(face, str(count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face Not Found")

    # Break loop when Enter (13) is pressed or 100 images are captured
    if cv2.waitKey(1) == 13 or count >= 100:
        break

# Release camera and close all windows
cap.release()
cv2.destroyAllWindows()
print("Dataset collection completed!")




