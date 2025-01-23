import sounddevice as sd
from scipy.io.wavfile import write
import os
from datetime import datetime

# 녹음 설정
samplerate = 16000  # 샘플링 레이트
duration = 5  # 녹음 시간 (초)
output_folder = "hjm"  # 녹음 파일을 저장할 폴더

# 폴더가 없는 경우 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 녹음 반복 횟수
num_recordings = 10  # 원하는 녹음 횟수 설정

for i in range(1, num_recordings + 1):
    # 고유한 파일명을 위해 날짜와 시간을 포함한 형식으로 지정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_folder, f"record_{timestamp}_{i:03}.wav")
    
    print(f"{i}/{num_recordings}: 녹음을 시작합니다... ({filename})")
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()  # 녹음이 끝날 때까지 대기
    write(filename, samplerate, recording)  # 녹음 파일 저장
    print(f"녹음이 완료되었습니다: {filename}")

print("모든 녹음이 완료되었습니다.")
