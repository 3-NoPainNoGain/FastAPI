import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU 사용 여부: {'사용' if torch.cuda.is_available() else '미사용'}")

# 데이터 경로 설정 (VS Code 기준 상대 경로)
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "datasets", "processed")

# 데이터 로드
X_orig = np.load(os.path.join(DATA_DIR, "X.npy"))
Y_orig = np.load(os.path.join(DATA_DIR, "Y.npy"))
X_aug = np.load(os.path.join(DATA_DIR, "augmented_X.npy"))
Y_aug = np.load(os.path.join(DATA_DIR, "augmented_Y.npy"))

# ✅ 클래스 불일치 확인 위치 
print("원본 클래스 목록:", set(np.unique(Y_orig)))
print("증강 클래스 목록:", set(np.unique(Y_aug)))

if set(np.unique(Y_orig)) != set(np.unique(Y_aug)):
    print("⚠️ 클래스가 일치하지 않습니다! 병합하기 전에 증강 데이터를 다시 확인하세요.")
    exit()  # 또는 raise ValueError("클래스 불일치!")

# 클래스 병합
X = np.concatenate([X_orig, X_aug], axis=0)
Y = np.concatenate([Y_orig, Y_aug], axis=0)
print(f"총 데이터 shape: {X.shape}, 총 라벨 shape: {Y.shape}")

# 정규화
scaler = StandardScaler()
X = X.reshape(-1, 258)
X = scaler.fit_transform(X)
X = X.reshape(-1, 30, 258)

# Train/Validation Split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# 텐서 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.long)

# 데이터로더
batch_size = 16
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

# 모델 정의
class SignLanguage1DCNN(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguage1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(258, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.activation(self.fc1(x)))
        return self.fc2(x)
    
    # 👇 load_model 함수 정의
    def load_model(model_path=None):
        BASE_DIR = os.path.dirname(__file__)
        DATA_DIR = os.path.join(BASE_DIR, "datasets", "processed")
        MODEL_PATH = model_path or os.path.join(BASE_DIR, "models", "aug_model_val_99.17_20250525.pth")

        classes = np.load(os.path.join(DATA_DIR, "classes.npy"), allow_pickle=True)
        model = SignLanguage1DCNN(num_classes=len(classes))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
        model.eval()
        return model

# 학습 준비
model = SignLanguage1DCNN(num_classes=len(np.unique(Y))).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 학습 함수
def train_model(model, train_loader, val_loader, epochs=30):
    best_accuracy = 0.0
    early_stop_count = 0
    patience = 10
    
    for epoch in range(epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct_train += (outputs.argmax(1) == Y_batch).sum().item()
            total_train += Y_batch.size(0)

        train_accuracy = 100 * correct_train / total_train

        # 검증
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, Y_batch).item()
                correct_val += (outputs.argmax(1) == Y_batch).sum().item()
                total_val += Y_batch.size(0)

        val_accuracy = 100 * correct_val / total_val
        print(f"[Epoch {epoch+1}/{epochs}] Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")

        scheduler.step()

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(DATA_DIR, "aug_model_val_99.17_20250525.pth"))
            print("⭐ Best model saved")
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print("⛔ Early stopping")
                break

# 학습 시작
train_model(model, train_loader, val_loader)

# 최종 정확도 출력
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", "aug_model_val_99.17_20250525.pth")))
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, Y_batch in val_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        outputs = model(X_batch)
        correct += (outputs.argmax(1) == Y_batch).sum().item()
        total += Y_batch.size(0)

print(f"최종 Validation Accuracy: {100 * correct / total:.2f}%")

# 클래스 정보 로드
CLASSES_PATH = os.path.join(DATA_DIR, "classes.npy")
if os.path.exists(CLASSES_PATH):
    classes = np.load(CLASSES_PATH, allow_pickle=True)
else:
    raise FileNotFoundError("classes.npy 파일이 datasets/processed/ 경로에 없습니다.")

# 예측 함수
def predict_from_keypoints(keypoints_30x258, model=None):
    if model is None:
        model = SignLanguage1DCNN(num_classes=len(classes))
        model.load_state_dict(torch.load(os.path.join(DATA_DIR, "aug_model_val_99.17_20250525.pth"), map_location=device))
        model.eval()

    if isinstance(keypoints_30x258, list):
        keypoints_30x258 = np.array(keypoints_30x258)

    if keypoints_30x258.shape != (30, 258):
        raise ValueError(f"❗ 입력 shape 오류: 기대값은 (30, 258), 현재는 {keypoints_30x258.shape}")

    input_tensor = torch.tensor(keypoints_30x258, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        return classes[pred_idx]
    
    # 👇 모델 불러오기 함수
def load_model(model_path=None):
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, "datasets", "processed")
    MODEL_PATH = model_path or os.path.join(BASE_DIR, "models", "aug_model_val_99.17_20250525.pth")

    classes = np.load(os.path.join(DATA_DIR, "classes.npy"), allow_pickle=True)

    model = SignLanguage1DCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model


# 👇 예측 함수
def predict_from_keypoints(keypoints_30x258, model=None):
    if model is None:
        model = load_model()

    if isinstance(keypoints_30x258, list):
        keypoints_30x258 = np.array(keypoints_30x258)

    if keypoints_30x258.shape != (30, 258):
        raise ValueError(f"❗ 입력 shape 오류: 기대값은 (30, 258), 현재는 {keypoints_30x258.shape}")

    input_tensor = torch.tensor(keypoints_30x258, dtype=torch.float32).unsqueeze(0)  # (1, 30, 258)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()

        classes = np.load(os.path.join(os.path.dirname(__file__), "datasets", "processed", "classes.npy"), allow_pickle=True)
        return classes[pred_idx]
