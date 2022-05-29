import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
from tensorflow.keras.models import load_model


def mediapipe_detection(image, model):
    #image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def extract_keypoints(results):
    pose = np.array([])
    if results.pose_landmarks:
        for i, res in enumerate(results.pose_landmarks.landmark):
            if i >= 25:
                break
            pose = np.append(pose, [res.x, res.y, res.z, res.visibility])
    else:
        pose = np.zeros((25*4))
    pose = pose.flatten()
    
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    #print(pose.shape)
    #print(lh.shape)
    #print(rh.shape)
    return np.concatenate([pose, lh, rh])

def download_video(video_url) :
    os.makedirs("video", exist_ok=True)
    time_str = time.strftime("%m%d_%H%M%S", time.localtime(time.time()))
    download_video = f'video/download_{time_str}.mp4'
    urllib.request.urlretrieve(video_url, download_video)
    return download_video

    
def predict_video(video_url):
    # 액션
    actions = np.array(['아버지', '어머니', '부모님', '대한민국', '기술', 
                        '교육', '대학교', '교수', '졸업', '도서관', 
                        '화장실', '컴퓨터', '안녕하세요', '고맙습니다', '기쁘다', 
                        '슬프다', '개', '고양이', '토끼', '수어'])
    #actions = np.array(['교육', '대학교', '교수', '졸업', '도서관'])

    # 동영상 길이(프레임 90)
    video_length = 90

    mp_holistic = mp.solutions.holistic # Holistic model
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 모델 로드
    model = load_model('model\model_0528_1928.h5')

    cap = cv2.VideoCapture(video_url)

    sequence = []
    for frame_num in range(video_length):
        ret, frame = cap.read()

        if frame_num % 2 == 1:
            continue
        frame_num2 = frame_num // 2

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        #print(results)

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
    cap.release()
    holistic.close

    res = model.predict(np.expand_dims(sequence, axis=0))[0]
    print(actions[np.argmax(res)], res)

    return actions[np.argmax(res)]