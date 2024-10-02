# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import os

import sys
import argparse

import time

import cv2
import numpy as np
import cv2 as cv

from sface import SFace

from yunet import YuNet

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
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)
        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)

    return output


# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(
    description="SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition (https://ieeexplore.ieee.org/document/9318547)")
parser.add_argument('--input1', '-i1', type=str,
                    help='Usage: Set path to the input image 1 (original face).')
parser.add_argument('--input2', '-i2', type=str,
                    help='Usage: Set path to the input image 2 (comparison face).')
parser.add_argument('--model', '-m', type=str, default='face_recognition_sface_2021dec.onnx',
                    help='Usage: Set model path, defaults to face_recognition_sface_2021dec.onnx.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--dis_type', type=int, choices=[0, 1], default=0,
                    help='Usage: Distance type. \'0\': cosine, \'1\': norm_l1. Defaults to \'0\'')
args = parser.parse_args()

if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]
    # Instantiate SFace for face recognition
    recognizer = SFace(modelPath=args.model,
                       disType=args.dis_type,
                       backendId=backend_id,
                       targetId=target_id)
    # Instantiate YuNet for face detection
    detector = YuNet(modelPath='face_detection_yunet_2023mar.onnx',
                     inputSize=[320, 320],
                     confThreshold=0.9,
                     nmsThreshold=0.3,
                     topK=5000,
                     backendId=backend_id,
                     targetId=target_id)

    # # Detect faces
    # detector.setInputSize([img1.shape[1], img1.shape[0]])
    # face1 = detector.infer(img1)
    # assert face1.shape[0] > 0, 'Cannot find a face in {}'.format(args.input1)
    # detector.setInputSize([img2.shape[1], img2.shape[0]])
    # face2 = detector.infer(img2)
    # assert face2.shape[0] > 0, 'Cannot find a face in {}'.format(args.input2)
    #
    # # Match
    # result = recognizer.match(img1, face1[0][:-1], img2, face2[0][:-1])
    # #print('Result: {}.'.format('same identity' if result else 'different identities'))
    # print(result)

# camera demo
    folder_path = './Member_img/'
    
    member_name = ""
    time_checkin = 0
    max_dis = 0
    
    
    deviceId = 0
    
    countFrame = 0
    
    cap = cv.VideoCapture(deviceId)

    while True:
        hasFrame, frame = cap.read()
    
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        detector.setInputSize([w, h])
        
        # Inference
        face_cam = detector.infer(frame)  # face_cam là một tuple
        
        # Tính khoảng cách
        if face_cam is not None and len(face_cam) > 0 and countFrame < 20:
            countFrame += 1
            cv.putText(frame, 'Vui long cho', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            frame = visualize(frame, face_cam)
            cv.imshow('ShowCam', frame)
        
        if(face_cam is not None and len(face_cam) > 0 and countFrame == 20):
            cv.putText(frame, 'Dang xac minh', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            frame = visualize(frame, face_cam)
            cv.imshow('ShowCam', frame)
            
            member_name = ""
            max_dis = 0
            
            for filename in os.listdir(folder_path):
                folder_path_2 = folder_path + filename
                
                for filename_2 in os.listdir(folder_path_2):
                    img_data = cv.imread(folder_path_2 + "/" + filename_2)
                    detector.setInputSize([img_data.shape[1], img_data.shape[0]])
                    face_data = detector.infer(img_data)
                    
                    distance = recognizer.match(frame, face_cam[0][:-1], img_data, face_data[0][:-1])
                    
                    if(distance[0] > max_dis):
                        max_dis = distance[0]
                        member_name = filename
        
            if(max_dis > 0.5):
                print(member_name)
                time_checkin = time.time
                # Lưu member_name và time_checkin vào database mongo 
                
            countFrame = 0
            # Lưu
            # Chờ 10 giây cho lần chạy while tiếp theo
            time.sleep(4)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                        
                
                
            #     cv.putText(frame, 'Da phat hien khuon mat', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            #     distance = recognizer.match(frame, face_cam[0][:-1], img_ogn, face_ogn[0][:-1])
            #     print(f"Khoảng cách: {distance}")
            # else:
            #     cv.putText(frame, 'Chua phat hien khuon mat', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            #     print("Không phát hiện khuôn mặt trong khung hiện tại. Tiếp tục...")

            # # Vẽ kết quả lên hình ảnh
            # frame = visualize(frame, face_cam)

            # # Hiển thị kết quả
            # cv.imshow('ShowCam', frame)

            # Kiểm tra nếu người dùng nhấn 'Esc' để thoát
    #     if cv.waitKey(1) == 27:  # 27 là mã ASCII của phím Esc
    #         print("Thoát chương trình...")
    #         break

    # Giải phóng tài nguyên
    cap.release()
    cv.destroyAllWindows()


