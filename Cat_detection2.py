from flask import Flask, render_template, Response, request
import cv2
from threading import Thread, Lock
import time
import numpy as np

app = Flask(__name__)

stream_url = 'http://192.168.137.24:4747/video'
cap = cv2.VideoCapture(stream_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

width, height = 320, 240
video_frame = None
lock = Lock()

net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

target_classes = {"cat": (0, 255, 0, 0), "person": (255, 0, 0, 0), "tvmonitor": (0, 0, 255, 0), "bird": (0, 0, 0, 255)}

def camera_stream():
    global cap, video_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                if label in target_classes:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    startX, startY, endX, endY = max(0, startX), max(0, startY), min(w, endX), min(h, endY)
                    
                    color = target_classes[label]
                    text = "{}: {:.2f}%".format(label, confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        with lock:
            video_frame = frame.copy()

def gen_frames():
    global video_frame
    while True:
        with lock:
            if video_frame is None:
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode('.jpg', video_frame)
            if not ret:
                continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index_flask_server.html')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'

if __name__ == '__main__':
    camera_thread = Thread(target=camera_stream)
    camera_thread.start()
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        if cap.isOpened():
            cap.release()
