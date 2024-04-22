from flask import Flask, render_template, Response, request
import cv2
import numpy as np

app = Flask(__name__)

camera_id = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_camera', methods=['POST'])
def select_camera():
    global camera_id
    camera_id = int(request.form.get('camera'))
    return render_template('index.html')

def gen():
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        yield "Camera không tồn tại"
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: failed to capture image")
            break

        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
