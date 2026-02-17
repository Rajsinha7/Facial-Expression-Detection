# Facial-Expression-Detection
The Facial Expression Detection System is a real-time computer vision application developed using Python, OpenCV, and MediaPipe.
The system detects human faces from live video, extracts facial landmarks using MediaPipe Face Mesh, and classifies facial expressions into Happy and Neutral emotions based on the Mouth Aspect Ratio (MAR).

This project demonstrates practical implementation of facial landmark detection, emotion classification, and real-time video processing.

--> Features
- Real-time face detection using camera
- Facial landmark detection with MediaPipe Face Mesh
- Emotion classification (Happy / Neutral)
- Mouth Aspect Ratio (MAR) calculation
- Fast and efficient real-time processing
- Lightweight and beginner-friendly approach

--> Tech Stack
Programming Language
Python
Libraries
OpenCV – Computer vision and image processing
MediaPipe – Facial landmark detection (Face Mesh)
NumPy – Numerical computations
Tools & Models
Haar Cascades – Face detection
MediaPipe Face Mesh – Landmark extraction

--> Project Structure
facial-expression-detection/
│
├── haarcascade_frontalface_default.xml
├── facial_expression_detection.py
├── requirements.txt
└── README.md




--> How to Run the Project
1️. Prerequisites
Python 3.x installed
A working webcam

2️. Install Dependencies
pip install opencv-python mediapipe numpy

3️. Clone the Repository
git clone [https://github.com/<your-github-username>/facial-expression-detection.git](https://github.com/Rajsinha7/Facial-Expression-Detection/tree/main)
cd facial-expression-detection

4️. Run the Application
python facial_expression_detection.py

5️. Output
Webcam window opens
Face detected in real time
Facial landmarks displayed
Emotion label shown as Happy or Neutral
Press q to exit

--> How It Works
Webcam captures live video frames
Haar Cascade detects faces
MediaPipe Face Mesh extracts facial landmarks
Mouth Aspect Ratio (MAR) is calculated

Expression is classified based on MAR threshold
