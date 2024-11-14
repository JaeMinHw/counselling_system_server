import os
import asyncio
import json
import io
from websockets import serve
from pydub import AudioSegment
import speech_recognition as sr
import joblib
import numpy as np
import librosa

# 학습된 화자 구분 모델 로드
model = joblib.load("voice_recognition_model_binary.pkl")

# 음성 특징 추출 함수
def extract_features(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=512)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    features = np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(spectral_contrast, axis=1),
        np.mean(chroma, axis=1),
        np.mean(tonnetz, axis=1)
    ])
    return features

# 텍스트 추출 및 화자 구분 함수
def identify_speaker_and_transcribe(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="ko-KR")
        except sr.UnknownValueError:
            text = ""
    
    # 화자 구분
    features = extract_features(audio_file_path).reshape(1, -1)
    speaker_label = model.predict(features)[0]
    speaker = "me" if speaker_label == 1 else "another"
    return {"speaker": speaker, "text": text}

# WebSocket 연결 처리 함수
async def handle_connection(websocket, path):
    async for audio_data in websocket:
        try:
            print(f"Received audio data of length: {len(audio_data)}")

            # 오디오 데이터를 메모리 내에서 변환 및 WAV 저장
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export("temp_audio.wav", format="wav")

            # 화자 구분 및 텍스트 추출
            result = identify_speaker_and_transcribe("temp_audio.wav")

            # 결과 전송
            await websocket.send(json.dumps(result))
            print(f"Sent transcription result: {result}")

        except Exception as e:
            print(f"Error processing audio data: {e}")
            await websocket.send(json.dumps({"error": str(e)}))

# WebSocket 서버 시작
async def main():
    async with serve(handle_connection, "localhost", 8765):
        print("WebSocket server started...")
        await asyncio.Future()  # 서버 계속 실행

# 서버 실행
asyncio.run(main())
