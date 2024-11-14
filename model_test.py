import asyncio
import websockets
import tensorflow as tf
import numpy as np
import librosa
import joblib
import io
import os
import json
from pydub import AudioSegment
from google.cloud import speech
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "c:/Users/User/stone-climate-441309-e7-69ab160b40d4.json"
# Google Cloud Speech-to-Text 클라이언트 설정
speech_client = speech.SpeechClient()

# 모델 및 스케일러 로드
model = tf.keras.models.load_model("cnn_lstm_autoencoder_me.h5")
scaler = joblib.load("scaler_me.pkl")

# 음성 특징 추출 함수
def extract_mfcc(audio_data):
    # WebM 데이터를 WAV 형식으로 변환
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
    audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)  # 16-bit PCM으로 설정

    # WAV 형식으로 변환된 데이터를 librosa에서 처리 가능하게 준비
    audio_wav = io.BytesIO()
    audio.export(audio_wav, format="wav")
    audio_wav.seek(0)

    # 특징 추출
    y, sr = librosa.load(audio_wav, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# 화자 예측 함수
def predict_voice(audio_data, threshold=2.0):
    features = extract_mfcc(audio_data).reshape(1, -1)
    features = scaler.transform(features)

    reconstructed = model.predict(features)
    mse = np.mean((features - reconstructed) ** 2)

    if(mse < threshold) :
        label = "me"
    else :
        label = "another"
    return label, float(mse)

# 음성 텍스트 변환 함수
def transcribe_audio(audio_data):
    # WebM 데이터를 WAV로 변환하고 16-bit PCM 설정
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
    audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)  # 16-bit PCM으로 변환

    # Google Speech-to-Text API에서 처리할 수 있는 형식으로 변환
    audio_wav = io.BytesIO()
    audio.export(audio_wav, format="wav", parameters=["-acodec", "pcm_s16le"])  # 16-bit PCM 보장
    audio_wav.seek(0)

    # Google Speech-to-Text API 요청
    audio_content = audio_wav.read()
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR"
    )

    response = speech_client.recognize(config=config, audio=audio)
    if response.results:
        return response.results[0].alternatives[0].transcript
    else:
        return ""

# 웹소켓 서버
async def handle_connection(websocket, path):
    async for message in websocket:
        try:
            # 수신된 오디오 데이터 처리
            audio_data = message
            print("Received audio data")

            # 화자 예측 수행
            label, mse = predict_voice(audio_data, threshold=2.0)

            # 텍스트 변환 수행
            transcript = transcribe_audio(audio_data)

            # 결과 전송
            response = {
                "label": label,
                "mse": mse,
                "text": transcript
            }
            await websocket.send(json.dumps(response))
            print(f"Sent result: {response}")

        except Exception as e:
            print(f"Error processing audio data: {e}")
            await websocket.send(json.dumps({"error": str(e)}))

# 웹소켓 서버 시작
async def main():
    async with websockets.serve(handle_connection, "localhost", 8765):
        print("WebSocket server started...")
        await asyncio.Future()  # 서버 계속 실행

asyncio.run(main())
