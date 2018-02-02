from statistics import mode

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        self.video.set(3, 640)
        self.video.set(4, 480)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

        # parameters for loading data and images
        self.detection_model_path = os.path.dirname(__file__) + '/../trained_models/detection_models/haarcascade_frontalface_default.xml'
        self.emotion_model_path = os.path.dirname(__file__) + '/../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
        self.emotion_labels = get_labels('fer2013')

        # hyper-parameters for bounding boxes shape
        self.frame_window = 10
        self.emotion_offsets = (20, 40)

        # loading models
        self.face_detection = load_detection_model(self.detection_model_path)
        self.emotion_classifier = load_model(self.emotion_model_path, compile=False)

        # getting input model shapes for inference
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]

        # starting lists for calculating modes
        self.emotion_window = []

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def face_camera(self):
        bgr_image = self.video.read()[1]
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = detect_faces(self.face_detection, gray_image)

        for face_coordinates in faces:

            x1, x2, y1, y2 = apply_offsets(face_coordinates,
                                            self.emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (self.emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = self.emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = self.emotion_labels[emotion_label_arg]
            self.emotion_window.append(emotion_text)

            if len(self.emotion_window) > self.frame_window:
                self.emotion_window.pop(0)
            try:
                emotion_mode = mode(self.emotion_window)
            except:
                continue

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_mode, color, 0,
                        -45, 1, 1)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        jpeg = cv2.imencode('.jpg', bgr_image)[1]
        return jpeg.tobytes()
