import os
import numpy as np
import torch
from .model import SignLanguageBiLSTM

# 사용할 디바이스 설정 (GPU가 있으면 GPU 사용, 없으면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 현재 파일의 디렉토리 경로를 기준으로 모델과 클래스 경로 설정
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_bilstm_val_100_20250728.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "datasets", "classes.npy")

# 클래스 이름들을 numpy 배열 형태로 로드 (ex: ["안녕하세요", "열", "있다", ...])
classes = np.load(CLASSES_PATH, allow_pickle=True)

def load_model():
    """
    학습된 BiLSTM 모델을 로드하고 evaluation 모드로 설정하여 반환한다.
    모델은 클래스 개수에 맞춰 초기화되며, GPU 또는 CPU에 자동 할당된다.
    """
    model = SignLanguageBiLSTM(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # 평가 모드로 설정 (dropout, batchnorm 등이 inference로 작동)
    return model, classes

def predict_from_keypoints(keypoints_30x258, model, classes):
    """
    30프레임 분량의 keypoint 데이터를 받아 예측된 단어(class label)를 반환한다.

    Parameters:
        keypoints_30x258: (30, 258) shape의 numpy 배열 또는 list. 
                          각 프레임에서 추출된 keypoints가 시간 순서대로 쌓인 형태.
        model: 로드된 BiLSTM 모델
        classes: 단어 클래스 리스트 (numpy 배열)

    Returns:
        예측된 단어 클래스 (ex: "안녕하세요", "열", ...)
    """
    # 리스트로 들어온 경우 numpy 배열로 변환
    if isinstance(keypoints_30x258, list):
        keypoints_30x258 = np.array(keypoints_30x258)

    # 입력 형식이 정확히 (30, 258)인지 확인
    if keypoints_30x258.shape != (30, 258):
        raise ValueError(f"입력 shape 오류: 기대값은 (30, 258), 현재는 {keypoints_30x258.shape}")

    # 모델 입력을 위한 텐서 변환 및 차원 추가 (batch dimension)
    input_tensor = torch.tensor(keypoints_30x258, dtype=torch.float32).unsqueeze(0).to(device)

    # 예측 수행 (gradient 계산 비활성화)
    with torch.no_grad():
        output = model(input_tensor)  # 출력 shape: (1, num_classes)
        pred_idx = output.argmax(dim=1).item()  # 가장 확률 높은 클래스의 index 추출
        return classes[pred_idx]  # 예측된 클래스 이름 반환
