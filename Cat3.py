from flask import Flask, render_template, Response
import cv2
from threading import Thread, Lock
import time
import numpy as np
import os

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# URL của luồng video
stream_url = 'http://192.168.137.24:4747/video'
cap = cv2.VideoCapture(stream_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

video_frame = None
lock = Lock()

# Biến điều khiển ghi video
recording = False
video_writer = None
last_detected_time = 0  # Lưu thời gian phát hiện mèo

# Tạo thư mục lưu trữ ảnh và video nếu chưa tồn tại
if not os.path.exists("Picture,video"):
    os.makedirs("Picture,video")

# Tải mô hình MobileNet SSD đã được huấn luyện trước
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def camera_stream():
    global cap, video_frame, recording, video_writer, last_detected_time
    sensitivity = 0.4  # Điều chỉnh độ nhạy phát hiện mèo
    stop_record_after = 3  # Dừng ghi sau 3 giây nếu không còn phát hiện mèo
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        cat_detected = False
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > sensitivity:
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] == "cat":
                    cat_detected = True
                    last_detected_time = time.time()  # Cập nhật thời gian phát hiện mèo
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Chụp ảnh khi phát hiện mèo
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = os.path.join("Picture,video", f"cat_detected_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"Ảnh đã lưu: {filename}")

        # Kiểm tra trạng thái ghi video
        if cat_detected and not recording:
            recording = True
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            video_filename = os.path.join("Picture,video", f"cat_detected_{timestamp}.avi")
            video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 10, (w, h))
            print(f"Bắt đầu ghi video: {video_filename}")

        elif not cat_detected and recording:
            elapsed_time = time.time() - last_detected_time
            if elapsed_time >= stop_record_after:  # Kiểm tra nếu mèo biến mất hơn 3 giây
                recording = False
                if video_writer:
                    video_writer.release()
                    video_writer = None
                    print("Dừng ghi video do không phát hiện mèo sau 3 giây.")

        if recording and video_writer:
            video_writer.write(frame)
            cv2.putText(frame, "Đang ghi video...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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

if __name__ == '__main__':
    camera_thread = Thread(target=camera_stream)
    camera_thread.start()
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        if cap.isOpened():
            cap.release()
        if video_writer:
            video_writer.release()
