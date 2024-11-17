import asyncio
import websockets
import tensorflow as tf
import numpy as np
import librosa
import joblib
import io
import json
from pydub import AudioSegment
from google.cloud import speech
import base64
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "c:/Users/User/stone-climate-441309-e7-69ab160b40d4.json"
# Google Speech-to-Text 클라이언트 설정
speech_client = speech.SpeechClient()

# 모델 및 스케일러 로드
model = tf.keras.models.load_model("cnn_lstm_autoencoder_me.h5")
scaler = joblib.load("scaler_me.pkl")

# 음성 특징 추출 함수
def extract_mfcc(audio_data):
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
    audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)

    audio_wav = io.BytesIO()
    audio.export(audio_wav, format="wav")
    audio_wav.seek(0)

    y, sr = librosa.load(audio_wav, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# 화자 예측 함수
def predict_voice(audio_data, threshold=2.0):
    features = extract_mfcc(audio_data).reshape(1, -1)
    features = scaler.transform(features)

    reconstructed = model.predict(features)
    mse = np.mean((features - reconstructed) ** 2)
    print(mse)

    label = "me" if mse < threshold else "another"
    return label, float(mse)

# 텍스트 변환 함수
def transcribe_audio(audio_data):
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
    audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)

    audio_wav = io.BytesIO()
    audio.export(audio_wav, format="wav")
    audio_wav.seek(0)

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
            received_data = json.loads(message)
            encoded_audio = received_data.get("audio")
            sent_time = received_data.get("sentTime", "")

            if encoded_audio:
                audio_data = base64.b64decode(encoded_audio)

                label, mse = predict_voice(audio_data)
                transcript = transcribe_audio(audio_data)

                response = {
                    "label": label,
                    "text": transcript,
                    "start": 0.0,
                    "end": len(audio_data) / 16000,
                    "sentTime": sent_time
                }
                await websocket.send(json.dumps(response))
                print(f"Sent result: {response}")
            else:
                print("No audio data received.")

        except Exception as e:
            print(f"Error processing audio data: {e}")
            await websocket.send(json.dumps({"error": str(e)}))

# 웹소켓 서버 시작
async def main():
    async with websockets.serve(handle_connection, "localhost", 8765):
        print("WebSocket server started...")
        await asyncio.Future()

asyncio.run(main())