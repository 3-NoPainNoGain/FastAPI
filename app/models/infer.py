import numpy as np
import torch
from pathlib import Path
from .model import SignLanguageBiLSTM

ASSETS_DIR = Path(__file__).parent / "assets"
MODEL_PATH = ASSETS_DIR / "model_bilstm_val_100_20250728.pth"
CLASSES_PATH = ASSETS_DIR / "classes.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드 
def load_model():
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    model = SignLanguageBiLSTM(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() 
    return model, classes

def predict_from_keypoints(keypoints_30x258, model, classes):

    # 리스트로 들어온 경우 numpy 배열로 변환
    if isinstance(keypoints_30x258, list):
        keypoints_30x258 = np.array(keypoints_30x258)

    # 입력 형식이 정확히 (30, 258)인지 확인
    if keypoints_30x258.shape != (30, 258):
        raise ValueError(f"입력 shape 오류: 기대값은 (30, 258), 현재는 {keypoints_30x258.shape}")

    # 모델 입력을 위한 텐서 변환 및 차원 추가
    input_tensor = torch.tensor(keypoints_30x258, dtype=torch.float32).unsqueeze(0).to(device)

    # 예측 수행
    with torch.no_grad():
        output = model(input_tensor) 
        pred_idx = output.argmax(dim=1).item()  
    return classes[pred_idx]  
