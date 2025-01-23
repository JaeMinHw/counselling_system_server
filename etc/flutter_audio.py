# flutter과 파이썬과의 통신을 통해 오디오 저장하는 코드

import asyncio
import websockets
import io
from datetime import datetime
from pydub import AudioSegment
import os

# 오디오 파일을 저장할 폴더
output_folder = "jm"
os.makedirs(output_folder, exist_ok=True)

# WebSocket 연결 처리 함수
async def handle_connection(websocket, path):
    async for audio_data in websocket:
        try:
            
            # 현재 시간 기준으로 고유한 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_folder, f"audio_{timestamp}.wav")
            
            # WebSocket으로 받은 데이터가 WebM 형식이라면 변환
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
            audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)  # 16kHz, 16-bit, 모노 설정
            
            # 파일을 WAV 형식으로 저장
            audio.export(filename, format="wav")
            print(f"Audio file saved as {filename}")
            
        except Exception as e:
            print(f"Error processing audio data: {e}")

# WebSocket 서버 시작
async def main():
    async with websockets.serve(handle_connection, "localhost", 8765):
        print("WebSocket server started...")
        await asyncio.Future()  # 서버 계속 실행

# 서버 실행
asyncio.run(main())
