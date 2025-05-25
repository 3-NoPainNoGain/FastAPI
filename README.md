## 실행 방법

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/Scripts/activate 

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn main:app --reload
