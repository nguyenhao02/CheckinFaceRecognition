import os
import time
import cv2 
import numpy as np
from flask import Flask, jsonify
from sface import SFace  # import module SFace
from yunet import YuNet  # import module YuNet
from pymongo import MongoClient

app = Flask(__name__)

# Kết nối đến MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['face_recognition_db']
checkins = db['checkins']

# Hàm lưu vào MongoDB
def save_to_mongo(member_name, time_checkin):
    checkins.insert_one({
        'member_name': member_name,
        'time_checkin': time_checkin
    })
    
def visualize(image, face_cam, box_color=(0, 255, 0), text_color=(0, 0, 255)):
    output = image.copy()
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]

    for det in (face_cam if face_cam is not None else []):
        bbox = det[0:4].astype(np.int32)
        cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)
        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv2.circle(output, landmark, 2, landmark_color[idx], 2)

    return output

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA_FP16],
    [cv2.dnn.DNN_BACKEND_TIMVX,  cv2.dnn.DNN_TARGET_NPU],
    [cv2.dnn.DNN_BACKEND_CANN,   cv2.dnn.DNN_TARGET_NPU]
]

# Initialize models
recognizer = SFace(modelPath='face_recognition_sface_2021dec.onnx', disType=0, backendId=0, targetId=0)
detector = YuNet(modelPath='face_detection_yunet_2023mar.onnx', inputSize=[320, 320], confThreshold=0.9, nmsThreshold=0.3, topK=5000, backendId=0, targetId=0)

folder_path = './Member_img/'

@app.route('/start-recognition', methods=['POST'])
def start_recognition():
    
    member_name = ""
    time_checkin = 0
    max_dis = 0
    countFrame = 0
    # khởi động cam
    cap = cv2.VideoCapture(0)
    
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            return jsonify({'error': 'Cannot access camera'}), 500
    
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        detector.setInputSize([w, h])
        
        # Inference
        face_cam = detector.infer(frame)  # face_cam là một tuple
        
        # Tính khoảng cách
        if face_cam is not None and len(face_cam) > 0 and countFrame < 20:
            countFrame += 1
            cv2.putText(frame, 'Vui long cho', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            frame = visualize(frame, face_cam)
            cv2.imshow('ShowCam', frame)
        
        if face_cam is not None and len(face_cam) > 0 and countFrame == 20:
            cv2.putText(frame, 'Dang xac minh', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            frame = visualize(frame, face_cam)
            cv2.imshow('ShowCam', frame)
            
            member_name = ""
            max_dis = 0
            
            for filename in os.listdir(folder_path):
                folder_path_2 = folder_path + filename
                
                for filename_2 in os.listdir(folder_path_2):
                    img_data = cv2.imread(folder_path_2 + "/" + filename_2)
                    detector.setInputSize([img_data.shape[1], img_data.shape[0]])
                    face_data = detector.infer(img_data)
                    
                    distance = recognizer.match(frame, face_cam[0][:-1], img_data, face_data[0][:-1])
                    
                    if distance[0] > max_dis:
                        max_dis = distance[0]
                        member_name = filename
        
            if max_dis > 0.5:
                time_checkin = time.time()  # Gọi hàm để lấy giá trị thời gian hiện tại
                save_to_mongo(member_name, time_checkin)  # Lưu vào MongoDB
                cap.release()
                cv2.destroyAllWindows()
                return jsonify({'member_name': member_name, 'time_checkin': time_checkin})
                
            countFrame = 0
            # Chờ 4 giây cho lần chạy while tiếp theo
            time.sleep(4)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()  # Giải phóng camera
    cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ OpenCV
    return jsonify({'error': 'No face detected or matched'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
