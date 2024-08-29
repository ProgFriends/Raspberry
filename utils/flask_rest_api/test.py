# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run a Flask REST API exposing one or more YOLOv5s models with Webcam Streaming
"""

import argparse
import io
import threading

import torch
from flask import Flask, render_template, Response, request
from PIL import Image
import cv2
import time

import json
from flask_socketio import SocketIO, emit

import mysql.connector
from datetime import datetime

from flask import render_template

import tensorflow as tf
import tflite_runtime.interpreter as tflite
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

import time
import RPi.GPIO as GPIO

# MySQL 연결 설정
mysql_connection = mysql.connector.connect(
    host='127.0.0.1',
    #host='172.20.70.21',
    user='name',
    password='Biimeal@1234',
    database='ProgAr',
    autocommit=True
)

mysql_cursor = mysql_connection.cursor()

mysql_connection2 = mysql.connector.connect(
    host='15.165.56.246',
    user='name',
    password='Biimeal@1234',
    database='AndroidApp',
    autocommit=True
)

mysql_cursor2 = mysql_connection2.cursor()

app = Flask(__name__)

A1A = 24
A1B = 25

B1A = 27
B1B = 22

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)


GPIO.setup(A1A, GPIO.OUT)
GPIO.setup(A1B, GPIO.OUT)

GPIO.setup(B1A, GPIO.OUT)
GPIO.setup(B1B, GPIO.OUT)


socketio = SocketIO(app)
models = {}
video_capture = None

def reset_tables():
    while True:
        time.sleep(30)  # 5분마다 실행
        try:
            # leaf 테이블 초기화(자동승가속성 적용)
            truncate_query_leaf = "TRUNCATE TABLE leaf"
            mysql_cursor.execute(truncate_query_leaf)
            
            # master 테이블 초기화
            delete_query_master = "DELETE FROM master"
            mysql_cursor.execute(delete_query_master)
            
            mysql_connection.commit()
            print(f"{datetime.now()}: All records have been deleted from leaf and master tables.")
        except mysql.connector.Error as err:
            print(f"Error: {err}")

# 테이블 초기화하는 스레드(백그라운드에서 돌기)
reset_thread = threading.Thread(target=reset_tables)
reset_thread.daemon = True
reset_thread.start()
DETECTION_URL = '/v1/object-detection/<model>'

tflite_model_file = './utils/flask_rest_api/cnn_model.tflite'
interpreter = tf.lite.Interpreter(tflite_model_file)
interpreter.allocate_tensors()

current_pi = None
previous_pi = None
new_pi = None

current_class_name = None
last_coordinates_time = {}
last_coordinates= {}
last_seen_time= {}

CAPTURE_INTERVAL = 10  # in seconds
CAPTURE_FOLDER = 'captured_frames'

class_to_pi = {'leaf': 0, 'scindapsus': 1, 'hoya': 2, 'frydek': 3, 'corn': 4, 'monstera': 5, 'oak': 6, 'staghorn': 7, 'ivy': 8, 'plant': 9, 'sick_leaf' : 10}

def capture_frames():
    while True:
        time.sleep(CAPTURE_INTERVAL)
        ret, frame = video_capture.read()
        if not ret:
            break

        # Perform leaf detection on the captured frame
        detect_objects(models['yolov5s'], frame)


# hoya의 마지막 좌표 가져오는 함수
def get_last_hoya():
    query = "SELECT x1, y1, x2, y2 FROM master WHERE pn = 'hoya' ORDER BY created_at DESC LIMIT 1"
    mysql_cursor.execute(query)
    last_hoya = mysql_cursor.fetchone()
    return last_hoya

# pot의 마지막 좌표 가져오는 함수
def get_last_ivy():
    query = "SELECT x1, y1, x2, y2 FROM master WHERE pn = 'ivy' ORDER BY created_at DESC LIMIT 1"
    mysql_cursor.execute(query)
    last_ivy = mysql_cursor.fetchone()
    return last_ivy

def display_html_info(class_name, coordinates):
    global current_class_name
    
    a1, b1, a2, b2 = coordinates

    if class_name == 'hoya':
        name1_x = (a1 + a2) // 2
        name1_y = b1

        soil1_x = (a1 + a2) // 2
        soil1_y = b2
        
        if last_coordinates.get('hoya') != (name1_x, name1_y):
            socketio.emit('name1', {'display': True, 'x': name1_x, 'y': name1_y})
            socketio.emit('soil1', {'display': True, 'x': soil1_x, 'y': soil1_y})
            last_coordinates['hoya'] = (name1_x, name1_y, soil1_x, soil1_y)
           
            # MySQL에서 식물 이름 조회
            mysql_cursor2.execute("SELECT plantName FROM 3_plants WHERE plantSpecies = 'hoya'")
            plant_name = mysql_cursor2.fetchone()
            if plant_name:
                socketio.emit('pname1', {'display': True, 'plant_name': plant_name[0]})
            else:
                socketio.emit('pname1', {'display': True, 'plant_name': 'X'})
                
        last_seen_time['hoya'] = time.time()
        
    if class_name == 'ivy':
        name2_x = (a1 + a2) // 2
        name2_y = b1
        
        soil2_x = (a1 + a2) // 2
        soil2_y = b2

        if last_coordinates.get('ivy') != (name2_x, name2_y):
            socketio.emit('name2', {'display': True, 'x': name2_x, 'y': name2_y})
            socketio.emit('soil2', {'display': True, 'x': soil2_x, 'y': soil2_y})
            last_coordinates['ivy'] = (name2_x, name2_y, soil2_x, soil2_y)
            
            # MySQL에서 식물 이름 조회
            mysql_cursor2.execute("SELECT plantName FROM 3_plants WHERE plantSpecies = 'ivy'")
            plant_name = mysql_cursor2.fetchone()
            if plant_name:
                socketio.emit('pname2', {'display': True, 'plant_name': plant_name[0]})
            else:
                socketio.emit('pname2', {'display': True, 'plant_name': 'X'})
                
        last_seen_time['ivy'] = time.time()



def hide_html_info():
    current_time = time.time()
    hide_threshold = 2  # 객체가 보이지 않는 상태에서 2초 후에 HTML 요소를 숨김

    if 'hoya' in last_seen_time and current_time - last_seen_time['hoya'] > hide_threshold:
        socketio.emit('soil1', {'display': False})
        socketio.emit('name1', {'display': False})
        socketio.emit('pname1', {'display': False})
        del last_seen_time['hoya']

    if 'ivy' in last_seen_time and current_time - last_seen_time['ivy'] > hide_threshold:
        socketio.emit('soil2', {'display': False})
        socketio.emit('name2', {'display': False})
        socketio.emit('pname2', {'display': False})
        del last_seen_time['ivy']


def detect_objects(model, frame):
    global current_class_name
    
    results = model(frame, size=640)
    detections = results.xyxy[0].cpu().numpy()

    for detection in detections:
        class_index = int(detection[5])
        class_name = model.names[class_index]
        bbox = detection[:4].tolist()

        if class_name in class_to_pi and class_name != 'leaf':
            # 현재 클래스가 'hoya'인 경우
            if class_name == 'hoya':
                current_class_name = 'hoya'
            # 현재 클래스가 'ivy'인 경우
            elif class_name == 'ivy':
                current_class_name = 'ivy'

            # HTML 정보 표시 함수 호출
            display_html_info(class_name, bbox)

	
    hide_html_info()
    

    
    return results.pandas().xyxy[0].to_json(orient='records')
        

        #elif class_name == 'leaf' and current_pi is not None:
         #   x1, y1, x2, y2 = map(int, bbox)
         #   save_to_mysql(current_class_name, bbox, current_pi)
         #   leaf_coordinates.append(bbox)  # 리프 좌표 추가


def save_to_mysql(current_class_name, bbox, pi):
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')

    if pi is not None:
        # 중복 데이터 체크
        query_check_duplicate = "SELECT * FROM master WHERE pi = %s"
        values_check_duplicate = (pi,)
        mysql_cursor.execute(query_check_duplicate, values_check_duplicate)
        duplicate_data = mysql_cursor.fetchone()

        if duplicate_data:
            # 중복된 데이터인 pi가 있으면 환경정보 갱신
            query_update_env = "UPDATE master SET air = %s, soil = %s, created_at = %s WHERE pi = %s"
            values_update_env = (3, 3, current_time, pi)  # air과 soil은 임시값,, 이거 3이랑 5 뭔가 이상하게 저장되는거 같으니까 확인하기. 
            mysql_cursor.execute(query_update_env, values_update_env)
        else:
            # 중복된 데이터가 없으면 새로 추가, 인식 한번 됐을때만(leaf 하나 저장되어있을 떄만 이렇게 찍힘 )
            query_insert_data = "INSERT INTO master (pi, pn, air, soil, created_at,x1,y1,x2,y2) VALUES (%s, %s, %s, %s, %s,%s,%s,%s,%s)"
            values_insert_data = (current_pi, current_class_name, 5, 5, current_time,a1,b1,a2,b2)
            mysql_cursor.execute(query_insert_data, values_insert_data)    

        # 'leaf' 테이블에 정보 저장
        leaf_query = "INSERT INTO leaf (pi, x1, y1, x2, y2) VALUES (%s, %s, %s, %s, %s)" #감지된 pi로 들어옴 
        leaf_values = (pi, bbox[0], bbox[1], bbox[2], bbox[3])
        
        try:
            mysql_cursor.execute(leaf_query, leaf_values)
            mysql_connection.commit()
        except mysql.connector.Error as err:
            print(f"Error: {err}")
'''
def img_from_db(pi, mysql_cursor, interpreter):
    images = []
    li_list = []  # li 저장 리스트

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    try:
        query = "SELECT li, image FROM leaf WHERE pi = %s"
        mysql_cursor.execute(query, (pi,))
        results = mysql_cursor.fetchall()
        for result in results:
            image_li, image_data = result
            img = Image.open(io.BytesIO(image_data))
            img = img.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
            img_array /= 255.0  # 정규화
            images.append(img_array)
            li_list.append(image_li)
    except Exception as e:
        print(f"Error loading images from database: {e}")
    return images, li_list

def predict_img(images, li_list, mysql_cursor, interpreter):
    for img_array, image_li in zip(images, li_list):
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        predictions = output_data[0]  # 첫 번째 결과 추출
        predicted_index = np.argmax(predictions)
        one_hot_predictions = tf.one_hot(predicted_index, depth=len(predictions))
        save_prediction(one_hot_predictions.numpy(), image_li, mysql_cursor)  # li와 함께 저장
        print("One-hot 인코딩된 결과:", one_hot_predictions.numpy())
        
def save_prediction(prediction, image_li, mysql_cursor):
    is_healthy = int(prediction[5])  # 'healthy' 클래스의 예측 결과
    value_healthy = 0 if is_healthy == 1 else 1  # healthy가 참(1)일 경우에는 db에 0으로 저장, 그렇지 않을 경우에는 db에 1로 저장
    try:
        update_dis = "UPDATE leaf SET dis_ab = %s WHERE li = %s"
        mysql_cursor.execute(update_dis, (value_healthy, image_li))
        mysql_connection.commit() 
        print("Database updated successfully")
    except Exception as e:
        print(f"Error updating prediction in database: {e}")
'''

def video_stream():
    global video_capture
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.start()
    
    frame_count = 0
    frame_skip = 5  # 5 프레임마다 한 번씩 처리

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip == 0:
            for model_name, model in models.items():
                detect_objects(model, frame)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
          
def activate_motor2():
	GPIO.output(A1A, GPIO.HIGH)
	GPIO.output(A1B, GPIO.LOW)
	time.sleep(5)
	GPIO.output(A1A, GPIO.LOW)
	GPIO.output(A1B, GPIO.LOW)
	               
def activate_motor():
	GPIO.output(B1A, GPIO.HIGH)
	GPIO.output(B1B, GPIO.LOW)
	time.sleep(5)
	GPIO.output(B1A, GPIO.LOW)
	GPIO.output(B1B, GPIO.LOW)


@app.route('/')
def index():
	
	query_get_latest_data = "SELECT air, soil,tem FROM humidity ORDER BY id DESC LIMIT 1"
	mysql_cursor.execute(query_get_latest_data)
	latest_data = mysql_cursor.fetchone()
	air_hum = latest_data[0] if latest_data else None
	hum = latest_data[1] if latest_data else None
	air_tem = latest_data[2] if latest_data else None
	
	return render_template('index.html', air_hum=air_hum, hum=hum, air_tem=air_tem)
	
	#return render_template('index.html')

@app.route(DETECTION_URL, methods=['POST'])
def predict(model):
    if request.method != 'POST':
        return

    if request.files.get('image'):
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        if model in models:
            results = detect_objects(models[model], im)
            return results

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
                    
@app.route('/update_humidity', methods=['POST'])
def update_humidity():
	data = request.json
	air_hum = data.get('air_hum')
	hum = data.get('hum')
	air_tem = data.get('air_tem')
	hum2 = data.get('hum2')
	
	socketio.emit('humidity_update', {'air_hum': air_hum, 'hum': hum, 'air_tem': air_tem, 'hum2': hum2})
	return 'OK'
	
@app.route('/activate_motor', methods=['POST'])
def activate_motor_endpoint():
	if request.method == 'POST':
		activate_motor()
		return 'Motor activated'

@app.route('/activate_motor2', methods=['POST'])
def activate_motor2_endpoint():
	if request.method == 'POST':
		activate_motor2()
		return 'Motor activated'

	
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask API exposing YOLOv5 model with Webcam Streaming')
    parser.add_argument('--port', default=5000, type=int, help='port number')
    parser.add_argument('--model', nargs='+', default=['yolov5s'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()

    for m in opt.model:
        models[m] = torch.hub.load('ultralytics/yolov5', 'custom', './runs/Progplant6/weights/best.pt', force_reload=True, skip_validation=True)

    # OpenCV VideoCapture for webcam
    video_capture = cv2.VideoCapture(0)
 
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    video_capture.set(cv2.CAP_PROP_FPS,1)
    
    # 시작 시 클래스별로 마지막으로 좌표를 전송한 시간 초기화
    for class_name in models['yolov5s'].names:
        last_coordinates_time[class_name] = 0

    # Flask 앱 실행
    socketio.run(app, host='0.0.0.0', port=opt.port, use_reloader=False, debug=True)
