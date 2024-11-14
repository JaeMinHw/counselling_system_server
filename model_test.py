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
speech_client = speech.SpeechClient()

model = tf.keras.models.load_model("cnn_lstm_autoencoder_me.h5")
scaler = joblib.load("scaler_me.pkl")

def extract_mfcc(audio_data):
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
    audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)

    audio_wav = io.BytesIO()
    audio.export(audio_wav, format="wav")
    audio_wav.seek(0)

    y, sr = librosa.load(audio_wav, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def predict_voice(audio_data, threshold=2.0):
    features = extract_mfcc(audio_data).reshape(1, -1)
    features = scaler.transform(features)

    reconstructed = model.predict(features)
    mse = np.mean((features - reconstructed) ** 2)

    label = "me" if mse < threshold else "another"
    return label, float(mse)

def transcribe_audio(audio_data):
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
    audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)

    audio_wav = io.BytesIO()
    audio.export(audio_wav, format="wav", parameters=["-acodec", "pcm_s16le"])
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

async def handle_connection(websocket, path):
    async for message in websocket:
        try:
            # 메시지를 JSON으로 디코드하고 오디오 데이터와 전송 시간 분리
            data = json.loads(message)
            audio_data = data.get("audio")
            sent_time = data.get("sentTime", "unknown")

            if audio_data is None:
                await websocket.send(json.dumps({"error": "No audio data received"}))
                continue

            print("Received audio data")

            # 화자 예측
            label, mse = predict_voice(audio_data, threshold=2.0)

            # 텍스트 변환
            transcript = transcribe_audio(audio_data)

            # 결과 전송
            response = {
                "label": label,
                "mse": mse,
                "text": transcript,
                "sentTime": sent_time  # 전송 시간 추가
            }
            await websocket.send(json.dumps(response))
            print(f"Sent result: {response}")

        except Exception as e:
            print(f"Error processing audio data: {e}")
            await websocket.send(json.dumps({"error": str(e)}))

async def main():
    async with websockets.serve(handle_connection, "localhost", 8765):
        print("WebSocket server started...")
        await asyncio.Future()

asyncio.run(main())
