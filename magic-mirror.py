import subprocess
import speech_recognition as sr
from datetime import datetime
import pyttsx3
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Emotion detection setup
face_classifier = cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')
emotion_interpreter = Interpreter(model_path="/home/shadril238/Desktop/magic-mirror/emotion_detection_model_100epochs_no_opt.tflite")
emotion_interpreter.allocate_tensors()
emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start video capture
cap = cv2.VideoCapture(0)
prev_emotion = None

while True:

    # Emotion detection
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera. Check camera initialization.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        roi = roi_gray.astype('float32') / 255
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)

        emotion_interpreter.set_tensor(emotion_input_details[0]['index'], roi)
        emotion_interpreter.invoke()
        emotion_preds = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])

        emotion_label = class_labels[emotion_preds.argmax()]
        if emotion_label != prev_emotion:
            engine.say(f"Your expression seems {emotion_label.lower()}.")
            engine.runAndWait()
            prev_emotion = emotion_label

        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


