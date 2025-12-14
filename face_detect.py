import cv2
import numpy as np
import os

# Path to Haar cascade classifier for face detection
haar_file = "haarcascade_frontalface_default.xml"

# Path to the dataset (update this if your path is different)
datasets = "dataset"

# Initialize data structures
(images, labels, names, id) = ([], [], {}, 0)

# Load training images and labels
for root, dirs, files in os.walk(datasets):
    for dir_name in dirs:
        names[id] = dir_name  # e.g., names[0] = "person1"
        subject_path = os.path.join(datasets, dir_name)
        for filename in os.listdir(subject_path):
            path = os.path.join(subject_path, filename)
            img = cv2.imread(path, 0)  # Read in grayscale
            if img is None:
                print(f"Image {path} could not be read.")
                continue
            images.append(img)
            labels.append(id)
        id += 1
    break  # prevent descending into deeper subdirectories

# Convert to NumPy arrays
(images, labels) = [np.array(lists) for lists in [images, labels]]

# Check if training data is sufficient
print("Number of training images:", len(images))
print("Number of labels:", len(labels))

if len(images) < 2:
    print("❌ Not enough training data. You need at least 2 images.")
    exit()

# Create and train the face recognizer model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(haar_file)

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = img_gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (640, 480))

        prediction = model.predict(face_resize)
        confidence = prediction[1]  # lower = better match

        if confidence < 74:
            label = names[prediction[0]].strip()
        else:
            label = "Unknown"

        cv2.putText(frame, label, (x + 5, y + h + 25),
                    cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (0, 255, 0) if label != "Unknown" else (0, 0, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
