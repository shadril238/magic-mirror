import subprocess
import speech_recognition as sr
from datetime import datetime
import pyttsx3
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np
import time
import random

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

# Fancy phrases for each emotion
fancy_phrases = {
    'Angry': "You look quite intense. Is everything okay?",
    'Disgust': "That's a very strong expression. Did something unpleasant happen?",
    'Fear': "You seem a bit startled. Remember, it's going to be alright.",
    'Happy': "What a wonderful smile! Your joy is infectious.",
    'Neutral': "Keeping it cool and composed, I see. You have a calm presence.",
    'Sad': "It's okay to feel down sometimes. I'm here if you need to talk.",
    'Surprise': "Oh! Didn't see that coming, did you? You seem taken aback!"
}

def speak_emotion_with_phrase(emotion):
    # Speak the detected emotion
    engine.say(f"You seem {emotion.lower()}.")
    engine.runAndWait()
    # Delay before speaking the fancy phrase
    time.sleep(0.5)
    # Speak the fancy phrase
    engine.say(fancy_phrases[emotion])
    engine.runAndWait()
    # Introduce a delay after speaking to prevent constant repetition
    time.sleep(3)

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

        # Normalize pixel values and add batch dimension
        roi = roi_gray.astype('float32') / 255
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)

        # Make prediction
        emotion_interpreter.set_tensor(emotion_input_details[0]['index'], roi)
        emotion_interpreter.invoke()
        emotion_preds = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])

        # Get the label of the predicted emotion
        emotion_label = class_labels[emotion_preds.argmax()]
        # Only speak if the emotion has changed
        if emotion_label != prev_emotion:
            speak_emotion_with_phrase(emotion_label)
            prev_emotion = emotion_label

        # Display the label on the screen
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

