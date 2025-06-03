# handDoc - FastAPI
`FastAPI` 기반으로 실시간 WebSocket 통신을 처리하며, 프론트엔드로부터 영상 프레임을 수신한 후, 수어 인식 모델을 통해 결과를 반환합니다.

### 🛠️ 기술 스택

- Python : 전체 백엔드 로직 기반
- FastAPI : WebSocket 기반 비동기 서버 구현
- MediaPipe : 영상에서 수어 키포인트 추출
- NumPy / OpenCV : 프레임 처리 및 배열 변환
- PyTorch : 사전 학습된 수어 인식 모델 로드 및 추론
- Uvicorn : FastAPI 실행 서버

### 실행 방법

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/Scripts/activate 

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn main:app --reload
