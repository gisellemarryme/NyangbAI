# 구글 코랩에서 진행
from google.colab import drive
drive.mount('/content/drive')

# 깃 코드 클론해서 사용
!git clone https://github.com/ultralytics/yolov5.git

%cd yolov5
!pip install -r requirements.txt

import torch
if torch.cuda.is_available():
    device = torch.device('cuda')  # GPU 사용
    print(device)

# coco 데이터셋에서 0,1,2,3,5,7,9,11 피쳐만 인식
!python /content/yolov5/detect.py --weights '/content/drive/MyDrive/2024_2/MV/final_project/yolov5/weights/yolov5s.pt' --source '/content/drive/MyDrive/2024_2/MV/final_project/test_motor.mp4' --classes 0 1 2 3 5 7 9 11 --project /content/drive/MyDrive/yolo_output --name 'motor_test'
