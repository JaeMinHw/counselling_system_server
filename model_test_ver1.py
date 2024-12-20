import asyncio
import websockets
import tensorflow as tf
import numpy as np
import librosa
import joblib
import io
import json
from pydub import AudioSegment
import webrtcvad
import base64
import os
import time
from datetime import datetime
from google.cloud import speech



os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "c:/Users/User/stone-climate-441309-e7-69ab160b40d4.json"

# Google Speech-to-Text 클라이언트 설정
speech_client = speech.SpeechClient()

# 모델 및 스케일러 로드
# model = tf.keras.models.load_model("cnn_lstm_autoencoder_me2.h5")
# scaler = joblib.load("scaler_me2.pkl")
# threshold = 0.13

model = tf.keras.models.load_model("cnn_lstm_autoencoder_me3.h5")
scaler = joblib.load("scaler_me3.pkl")
threshold=0.083


output_folder = "jm"
os.makedirs(output_folder, exist_ok=True)  # 오디오 파일을 저장할 폴더

# 음성 특징 추출 함수
def extract_mfcc(audio_data):
    y, sr = librosa.load(io.BytesIO(audio_data), sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft = 1024)
    return np.mean(mfccs.T, axis=0)

# 화자 예측 함수
def predict_voice(audio_data):
    features = extract_mfcc(audio_data).reshape(1, -1)
    features = scaler.transform(features)

    reconstructed = model.predict(features)
    mse = np.mean((features - reconstructed) ** 2)
    print(f"MSE: {mse}")

    label = "me" if mse < threshold else "another"
    return label, float(mse)

# 텍스트 변환 함수
def transcribe_audio(audio_wav):
    audio_content = audio_wav.read()
    if not audio_content or len(audio_content) == 0:
        raise ValueError("Audio data is empty or invalid.")

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

# VAD로 음성 구간 탐지
def detect_voice_segments(audio_data):
    vad = webrtcvad.Vad()
    vad.set_mode(1)  # 가장 민감한 모드 =3 

    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
    audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)

    samples = audio.get_array_of_samples()
    sample_rate = audio.frame_rate
    chunk_duration = 0.03  # 30ms
    chunk_size = int(sample_rate * chunk_duration)
    segments = []

    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i + chunk_size]
        if len(chunk) < chunk_size:
            break

        is_speech = vad.is_speech(chunk.tobytes(), sample_rate)
        segments.append((i / sample_rate, (i + chunk_size) / sample_rate, is_speech))

    return segments

# 구간 기반 화자 판별 및 텍스트 변환
def process_audio(audio_data):
    voice_segments = detect_voice_segments(audio_data)
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
    audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)

    results = []
    current_segment = []

    for start, end, is_speech in voice_segments:
        if is_speech:
            current_segment.append((start, end))
        else:
            if current_segment:
                # 구간 병합
                segment_start = current_segment[0][0]
                segment_end = current_segment[-1][1]
                segment_audio = audio[segment_start * 1000:segment_end * 1000]

                # WAV 변환
                audio_wav = io.BytesIO()
                segment_audio.export(audio_wav, format="wav")
                audio_wav.seek(0)

                # 화자 판별 및 텍스트 변환
                label, mse = predict_voice(audio_wav.read())
                audio_wav.seek(0)  # 재설정
                text = transcribe_audio(audio_wav)

                if text:  # 텍스트가 감지된 경우에만 저장
                    results.append({
                        "label": label,
                        "mse": mse,  # MSE 추가
                        "text": text,
                        "start": segment_start,
                        "end": segment_end,
                    })

                current_segment = []

    return results

# 오디오 파일 저장 함수
def save_audio_file(audio_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 고유한 파일명 생성
    filename = os.path.join(output_folder, f"audio_{timestamp}.wav")
    
    # WebSocket으로 받은 데이터가 WebM 형식이라면 변환
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
    audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)  # 16kHz, 16-bit, 모노 설정
    
    # 파일을 WAV 형식으로 저장
    audio.export(filename, format="wav")
    print(f"Audio file saved as {filename}")
    return filename

# WebSocket 처리 함수
async def handle_connection(websocket, path):
    async for message in websocket:
        try:
            received_data = json.loads(message)
            encoded_audio = received_data.get("audio")
            sent_time = received_data.get("sentTime", "")

            if encoded_audio:
                # 만약 audio_data가 list 형태로 오면, 이를 bytes로 변환
                if isinstance(encoded_audio, list):
                    audio_data = bytes(encoded_audio)
                else:
                    # Base64 인코딩된 경우
                    audio_data = base64.b64decode(encoded_audio)

                # 동적 구간 생성 및 처리
                results = process_audio(audio_data)

                if results:  # 텍스트가 감지된 결과만 저장
                    # 오디오 파일 저장
                    save_audio_file(audio_data)

                    # 결과 전송
                    await websocket.send(json.dumps(results))
                    print(f"Sent results: {results}")
                else:
                    print("No text detected in the audio.")

            else:
                print("No audio data received.")

        except Exception as e:
            print(f"Error processing audio data: {e}")
            await websocket.send(json.dumps({"error": str(e)}))


# WebSocket 서버 시작
async def main():
    async with websockets.serve(handle_connection, "localhost", 8765, max_size= 2**25):
        print("WebSocket server started...")
        await asyncio.Future()  # 서버 계속 실행

# 서버 실행
asyncio.run(main())
