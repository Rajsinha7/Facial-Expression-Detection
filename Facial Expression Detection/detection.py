

import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar cascades
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Detect facial landmarks using Mediapipe
    results = face_mesh.process(rgb_frame)

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray_frame[y:y + h, x:x + w]

    # Loop through facial landmarks
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Extract specific landmarks for emotion detection
            # Example: Eyes (468-473) and Mouth (13, 14, 308, 78, 95)
            try:
                left_eye = np.array([(face_landmarks.landmark[468].x, face_landmarks.landmark[468].y),
                                     (face_landmarks.landmark[473].x, face_landmarks.landmark[473].y)])
            except IndexError:
                left_eye = np.array([])
            mouth = np.array([(face_landmarks.landmark[13].x, face_landmarks.landmark[13].y),
                              (face_landmarks.landmark[14].x, face_landmarks.landmark[14].y),
                              (face_landmarks.landmark[78].x, face_landmarks.landmark[78].y),
                              (face_landmarks.landmark[95].x, face_landmarks.landmark[95].y)])

            # Calculate mouth aspect ratio
            mouth_height = np.linalg.norm(mouth[0] - mouth[1])
            mouth_width = np.linalg.norm(mouth[2] - mouth[3])
            mouth_aspect_ratio = mouth_height / mouth_width

            # Determine emotion based on landmarks
            if mouth_aspect_ratio > 0.5:
                emotion = "Happy"
           

            else:
                emotion = "Neutral"

            cv2.putText(frame, emotion, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Emotion Detection with Mediapipe', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()



