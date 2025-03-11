# import cv2
# import numpy as np
# from os import listdir
# from os.path import isfile, join
#
# data_path = 'C:/Users/Omkari Sneha/OneDrive/Desktop/dataset/'
# onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
#
# Training_Data, Labels = [], []
#
# for i, files in enumerate(onlyfiles):
#     image_path = data_path + onlyfiles[i]
#     images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     Training_Data.append(np.asarray(images, dtype=np.uint8))
#     Labels.append(i)
# Labels = np.asarray(Labels, dtype=np.int32)
#
# model = cv2.face.LBPHFaceRecognizer_create()
#
# model.train(np.asarray(Training_Data), np.asarray(Labels))
#
# print("Dataset model training Completed")
#
# face_classifier = cv2.CascadeClassifier('C:/Users/Omkari Sneha/OneDrive/Desktop/haarcascade_frontalface_default.xml')
#
#
# def face_detector(img, size=0.5):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray, 1.3, 5)
#
#     if faces is():
#         return img,[]
#
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
#         roi = img[y:y + h, x:x + w]
#         roi = cv2.resize(roi,(200, 200))
#
#         return img, roi
#     cap = cv2.VideoCapture(0)
#     while True:
#
#         ret, frame = cap.read()
#         image, face = face_detector(frame)
#
#         try:
#             face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#             result = model.predict(face)
#
#             if result[1] < 500:
#                 confidence = int(100 * (1 - (result[1])/300))
#
#             if confidence > 82:
#                 cv2.putText(image, "Sneha", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#                 cv2.imshow('Face Cropper', image)
#
#             else:
#                 cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#                 cv2.imshow('Face Cropper', image)
#         except:
#             cv2.putText(image, "Face not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX,1, (255, 0, 0), 2)
#             cv2.imshow('Face Cropper', image)
#             pass
#         if cv2.waitKey(1) == 13:
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
import cv2
import numpy as np
import os

# Path to dataset
data_path = 'C:/Users/Omkari Sneha/OneDrive/Desktop/dataset/'

# Ensure dataset directory exists
if not os.path.exists(data_path):
    print("Error: Dataset directory not found!")
    exit()

# Load all image files
onlyfiles = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

# Prepare training data
Training_Data, Labels = [], []

for i, file in enumerate(onlyfiles):
    image_path = os.path.join(data_path, file)
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if images is None:
        print(f"Skipping corrupt image: {file}")
        continue

    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Convert Labels to NumPy array
Labels = np.asarray(Labels, dtype=np.int32)

# Check if dataset is empty
if len(Training_Data) == 0:
    print("Error: No valid images found for training!")
    exit()

# Initialize LBPH Face Recognizer
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))

print("✅ Dataset model training completed!")

# Load Haar Cascade Classifier for face detection
face_classifier = cv2.CascadeClassifier('C:/Users/Omkari Sneha/OneDrive/Desktop/haarcascade_frontalface_default.xml')


# Function to detect face
def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return img, None

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
        return img, roi

    return img, None


# Start video capture
cap = cv2.VideoCapture(0)

print("🎥 Starting real-time face recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        continue

    image, face = face_detector(frame)

    try:
        if face is not None:
            result = model.predict(face)
            confidence = int(100 * (1 - (result[1]) / 300))

            if confidence > 80:
                name = "Sneha"  # Change this to the actual person's name
                color = (255, 255, 255)
            else:
                name = "Unknown"
                color = (0, 0, 255)

            cv2.putText(image, name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        else:
            cv2.putText(image, "Face not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Face Recognizer', image)

    except Exception as e:
        print("Error during recognition:", e)
        pass

    # Press Enter (key 13) to exit
    if cv2.waitKey(1) == 13:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("🔴 Face recognition stopped!")

