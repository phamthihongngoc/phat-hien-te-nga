from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import threading
import requests
import time

app = Flask(__name__)

# --- Load mô hình ---
MODEL_PATH = "D:/KY2/AIoT/BTL/test/fall_detection_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# --- Cấu hình Telegram Bot ---
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

# --- Nhãn hành động ---
labels = ["Ngã về phía trước", "Ngã về phía sau", "Ngã sang trái", "Ngã sang phải", "Đứng bình thường"]

# --- Bộ nhớ đệm dữ liệu ---
window_size = 50
sensor_buffer = []
latest_action = "🚶‍♂️ Đứng bình thường"
sensor_data = {"time": [], "ax": [], "ay": [], "az": [], "gx": [], "gy": [], "gz": []}

def send_telegram_alert(message):
    """Gửi cảnh báo đến Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=data)

@app.route("/")
def home():
    """Hiển thị giao diện web"""
    return render_template("index.html")

@app.route("/sensor", methods=["POST"])
def receive_sensor_data():
    """Nhận dữ liệu từ ESP32 và dự đoán hành động"""
    global sensor_buffer, latest_action, sensor_data

    try:
        data = request.json
        ax, ay, az = data.get("ax"), data.get("ay"), data.get("az")
        gx, gy, gz = data.get("gx"), data.get("gy"), data.get("gz")

        if None in [ax, ay, az, gx, gy, gz]:
            return jsonify({"error": "Dữ liệu không hợp lệ"}), 400

        # Thêm dữ liệu vào bộ đệm
        sensor_buffer.append([ax, ay, az, gx, gy, gz])
        sensor_data["time"].append(time.time())
        sensor_data["ax"].append(ax)
        sensor_data["ay"].append(ay)
        sensor_data["az"].append(az)
        sensor_data["gx"].append(gx)
        sensor_data["gy"].append(gy)
        sensor_data["gz"].append(gz)

        # Giới hạn dữ liệu chỉ giữ 100 mẫu gần nhất
        for key in sensor_data:
            if len(sensor_data[key]) > 100:
                sensor_data[key].pop(0)

        # Khi đủ 50 mẫu, thực hiện dự đoán
        if len(sensor_buffer) == window_size:
            input_data = np.array(sensor_buffer).reshape(1, 50, 6)
            prediction = model.predict(input_data)
            predicted_label = np.argmax(prediction, axis=1)[0]
            latest_action = labels[predicted_label]

            # Nếu phát hiện té ngã, gửi cảnh báo Telegram
            if "Ngã" in latest_action:
                threading.Thread(target=send_telegram_alert, args=(f"🚨 Cảnh báo té ngã: {latest_action}",)).start()

            sensor_buffer.pop(0)  # Loại bỏ dữ liệu cũ nhất để giữ đúng kích thước

        return jsonify({"action": latest_action}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/latest_action", methods=["GET"])
def get_latest_action():
    """Trả về hành động mới nhất"""
    return jsonify({"action": latest_action})

@app.route("/sensor_data", methods=["GET"])
def get_sensor_data():
    """Trả về dữ liệu cảm biến mới nhất"""
    return jsonify(sensor_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

