SEQUENCE_LENGTH = 30   # 모델 입력 시퀀스 길이
WIN             = 6    # 최근 예측을 저장할 창 크기
MAJ             = 4    # 다수결 판정을 위한 임계값
INACTIVITY_SEC  = 1.2  # 비활성 상태로 간주할 시간 (초)
HARD_RESET_SEC  = 5.0  # 시스템을 초기화할 비활성 시간 (초)
COOLDOWN_SEC    = 1.0  # 동일 문장 반복 출력을 막기 위한 쿨다운
PAIR_MAX_BACK   = 6    # 동사-명사 조합을 찾을 최대 거리
RECV_TIMEOUT_SEC = 5.0 # WS 수신 타임아웃 (ping 유지용)