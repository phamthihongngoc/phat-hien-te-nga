const ctx = document.getElementById("sensorChart").getContext("2d");
const MAX_DATA_POINTS = 50; // Giới hạn số lượng điểm trên biểu đồ

const sensorChart = new Chart(ctx, {
    type: "line",
    data: {
        labels: [],
        datasets: [
            { label: "AccelX", borderColor: "red", backgroundColor: "rgba(255, 0, 0, 0.2)", borderWidth: 2, data: [], fill: true },
            { label: "AccelY", borderColor: "green", backgroundColor: "rgba(0, 255, 0, 0.2)", borderWidth: 2, data: [], fill: true },
            { label: "AccelZ", borderColor: "blue", backgroundColor: "rgba(0, 0, 255, 0.2)", borderWidth: 2, data: [], fill: true },
            { label: "GyroX", borderColor: "purple", backgroundColor: "rgba(128, 0, 128, 0.2)", borderWidth: 2, data: [], fill: true },
            { label: "GyroY", borderColor: "orange", backgroundColor: "rgba(255, 165, 0, 0.2)", borderWidth: 2, data: [], fill: true },
            { label: "GyroZ", borderColor: "pink", backgroundColor: "rgba(255, 192, 203, 0.2)", borderWidth: 2, data: [], fill: true },
        ],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        elements: {
            line: { tension: 0.3 } // Làm mượt đường biểu đồ
        },
        scales: {
            x: {
                title: { display: true, text: "Thời gian (giây)", font: { size: 14 } },
                ticks: { autoSkip: true, maxTicksLimit: 10, font: { size: 12 } }
            },
            y: {
                title: { display: true, text: "Giá trị cảm biến", font: { size: 14 } },
                beginAtZero: false,
                ticks: { font: { size: 12 } }
            }
        },
        animation: {
            duration: 300,
            easing: "easeInOutQuad"
        },
        plugins: {
            legend: { labels: { font: { size: 14 } } }
        }
    },
});

function updateAction() {
    fetch("/latest_action")
        .then(response => response.json())
        .then(data => {
            const actionElement = document.getElementById("action");
            actionElement.innerText = data.action;

            let bgColor = "white"; // Mặc định không có cảnh báo

            switch (data.action) {
                case "Ngã về phía trước":
                    actionElement.style.color = "#d32f2f"; // Đỏ
                    bgColor = "#ffebee"; // Nền cảnh báo đỏ nhạt
                    break;
                case "Ngã về phía sau":
                    actionElement.style.color = "#1976d2"; // Xanh dương
                    bgColor = "#e3f2fd"; // Nền cảnh báo xanh dương nhạt
                    break;
                case "Ngã sang trái":
                    actionElement.style.color = "#388e3c"; // Xanh lá
                    bgColor = "#e8f5e9"; // Nền cảnh báo xanh lá nhạt
                    break;
                case "Ngã sang phải":
                    actionElement.style.color = "#f57c00"; // Cam
                    bgColor = "#fff3e0"; // Nền cảnh báo cam nhạt
                    break;
                case "Đứng bình thường":
                    actionElement.style.color = "#333"; // Màu mặc định
                    break;
                default:
                    actionElement.style.color = "#777"; // Xám
            }

            document.getElementById("chartContainer").style.backgroundColor = bgColor;
        })
        .catch(error => console.error("Lỗi khi lấy dữ liệu:", error));
}

function updateSensorData() {
    fetch("/sensor_data")
        .then(response => response.json())
        .then(data => {
            const timeLabels = data.time.map(t => new Date(t * 1000).toLocaleTimeString());

            // Giữ số lượng điểm cố định, cuộn biểu đồ về bên phải
            if (sensorChart.data.labels.length >= MAX_DATA_POINTS) {
                sensorChart.data.labels.shift();
                sensorChart.data.datasets.forEach(dataset => dataset.data.shift());
            }

            sensorChart.data.labels.push(...timeLabels);
            sensorChart.data.datasets[0].data.push(...data.ax);
            sensorChart.data.datasets[1].data.push(...data.ay);
            sensorChart.data.datasets[2].data.push(...data.az);
            sensorChart.data.datasets[3].data.push(...data.gx);
            sensorChart.data.datasets[4].data.push(...data.gy);
            sensorChart.data.datasets[5].data.push(...data.gz);

            sensorChart.update();
        })
        .catch(error => console.error("Lỗi khi lấy dữ liệu cảm biến:", error));
}

// Cập nhật dữ liệu mỗi giây
setInterval(updateAction, 1000);
setInterval(updateSensorData, 1000);