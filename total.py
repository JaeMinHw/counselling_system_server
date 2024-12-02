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
from datetime import datetime
from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room
import threading
import random
import time
import pymysql
from datetime import datetime


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

clients = {}
threads = {}
running_threads = {}


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "c:/Users/User/stone-climate-441309-e7-69ab160b40d4.json"

# Google Speech-to-Text 클라이언트 설정
from google.cloud import speech
speech_client = speech.SpeechClient()

# 모델 및 스케일러 로드
model = tf.keras.models.load_model("cnn_lstm_autoencoder_me3.h5")
scaler = joblib.load("scaler_me3.pkl")
threshold=0.083

output_folder = "received_full_audio"
os.makedirs(output_folder, exist_ok=True)

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
    language_code="ko-KR",
    enable_word_time_offsets=True,  # 단어 시간 오프셋 활성화
    use_enhanced=True,              # 고급 인식 모델 사용
    )

    response = speech_client.recognize(config=config, audio=audio)
    if response.results:
        return response.results[0].alternatives[0].transcript
    else:
        return ""

# VAD로 음성 구간 탐지
def detect_voice_segments(audio_data):
    vad = webrtcvad.Vad()
    vad.set_mode(1)  # 가장 민감한 모드

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
def process_audio(audio_data,sent_time):
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

                results.append({
                    "label": label,
                    "mse": mse,  # MSE 추가
                    "text": text,
                    "start": segment_start,
                    "end": segment_end,
                    "send_time" : sent_time,
                })

                current_segment = []

    return results

# WebSocket 처리 함수
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
                    # Base64 인코딩된 경우, Base64 디코딩
                    try:
                        audio_data = base64.b64decode(encoded_audio)
                    except Exception as e:
                        raise ValueError("Base64 decoding failed: " + str(e))

                # 동적 구간 생성 및 처리
                results = process_audio(audio_data, sent_time)

                # 결과 전송
                await websocket.send(json.dumps(results))
                print(f"Sent results: {results}")
            else:
                print("No audio data received.")

        except Exception as e:
            print(f"Error processing audio data: {e}")
            await websocket.send(json.dumps({"error": str(e)}))



# WebSocket 처리 함수
async def full_handle(websocket, path):
    try:
        async for message in websocket:
            # JSON 데이터 수신
            data = json.loads(message)
            
            # 오디오 데이터와 전송 시간 추출
            audio_data = data.get("audio")  # 리스트 형태로 수신됨
            sent_time = data.get("sentTime")
            
            if not audio_data:
                await websocket.send(json.dumps({"status": "error", "message": "No audio data received"}))
                continue

            
            # 리스트 내부 데이터를 바이트로 병합
            try:
                audio_bytes = b"".join(bytes(chunk) for chunk in audio_data)
                print(f"Converted audio_bytes length: {len(audio_bytes)}")
            except Exception as e:
                raise ValueError(f"Error converting audio_data to bytes: {e}")

            # 오디오 데이터 저장
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"full_audio_{timestamp}.wav"

            filepath = os.path.join(output_folder, filename)
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
            audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)  # 16kHz, 16-bit, 모노 설정
            # 바이너리 데이터로 저장
            audio.export(filepath, format="wav")
            print(f"Full audio data received and saved to {filepath} at {sent_time}")
            
            # 성공 메시지 전송
            await websocket.send(json.dumps({"status": "success", "message": "Audio data saved successfully"}))
    except Exception as e:
        print(f"Error in full_handle: {e}")
        await websocket.send(json.dumps({"status": "error", "message": str(e)}))





# ------------------------------------------------------------------------------------------------------------------


def get_db_connection():
    return pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='1234',
        database='counselling_sys',
        cursorclass=pymysql.cursors.DictCursor
    )

def get_counselor_id_by_patient(patient_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT counsel_id FROM clients WHERE id = %s"
            cursor.execute(sql, (patient_id,))
            result = cursor.fetchone()
            if result:
                return result['counsel_id']
    finally:
        connection.close()
    return None

def get_patients_by_counselor(counselor_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT id FROM clients WHERE counsel_id = %s"
            cursor.execute(sql, (counselor_id,))
            result = cursor.fetchall()
            return [row['id'] for row in result]
    finally:
        connection.close()

def generate_data(room):
    while running_threads.get(room, False):
        data1 = random.uniform(0, 100)
        data2 = random.uniform(0, 100)
        data3 = random.uniform(0, 100)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Sending data to room {room}")
        socketio.emit('data_update', {'data1': data1, 'data2': data2, 'data3' : data3, 'time': current_time}, room=room)
        time.sleep(0.1)
    print(f"Data transmission stopped for room {room}")

def start_background_thread(room):
    running_threads[room] = True
    thread = threading.Thread(target=generate_data, args=(room,))
    threads[room] = thread
    thread.start()

@socketio.on('patient_login')
def handle_patient_login(data):
    patient_id = data['patient_id']
    counselor_id = get_counselor_id_by_patient(patient_id)
    
    if counselor_id is None:
        print(f"Could not find counselor_id for patient {patient_id}")
        return
    
    session_id = request.sid
    clients[session_id] = {
        'type': 'patient',
        'patient_id': patient_id,
        'counselor_id': counselor_id
    }
    
    room = counselor_id
    join_room(room)
    print(f"Patient {patient_id} joined room {room}")
    
    for client_sid, client_info in clients.items():
        if client_info['type'] == 'counselor' and client_info['counselor_id'] == counselor_id:
            socketio.emit('connection_request', {'message': f'Connect with patient {patient_id}?'}, room=client_sid)
            print(f"Prompted counselor {counselor_id} to connect with patient {patient_id}")
            break

@socketio.on('counselor_login')
def handle_counselor_login(data):
    counselor_id = data['counselor_id']
    
    session_id = request.sid
    clients[session_id] = {
        'type': 'counselor',
        'counselor_id': counselor_id
    }
    
    room = counselor_id
    join_room(room)
    print(f"Counselor {counselor_id} joined room {room}")
    
    patient_ids = get_patients_by_counselor(counselor_id)
    for patient_id in patient_ids:
        socketio.emit('connection_request', {'message': f'Connect with patient {patient_id}?'}, room=session_id)
        print(f"Counselor {counselor_id} prompted to connect with patient {patient_id}")

@socketio.on('accept_connection')
def handle_accept_connection():
    counselor_id = clients[request.sid]['counselor_id']
    room = counselor_id
    
    socketio.emit('connection_accepted', {'message': 'Connection accepted'}, room=room)
    print(f"Started data transmission for room {room}")


# 임의의 데이터 생성 및 전송
@socketio.on('start')
def handle_start():


    counsel_id = clients[request.sid]['counselor_id']
    room = counsel_id
    print(f"Start data transmission for room {room}")

    socketio.emit('sensor_start', room)

    
    # client_server에 전송

    



    # 이 코드는 임의의 데이터를 생성 후 바로 flutter에 전송
    
    
    if room in threads and not running_threads[room]:
        start_background_thread(room)
    elif room not in threads:
        start_background_thread(room)
    else:
        print(f"Data transmission already running for room {room}")

@socketio.on('stop')
def handle_stop():
    counsel_id = clients[request.sid]['counselor_id']
    room = counsel_id
    
    print(f"Stop data transmission for room {room}")
    if room in threads:
        running_threads[room] = False
        threads[room].join()
        del threads[room]
        del running_threads[room]
        print(f"Data transmission stopped for room {room}")

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    client_info = clients.pop(session_id, None)
    
    if client_info:
        room = client_info['counselor_id']
        leave_room(room)
        print(f"Client {session_id} left room {room}")
        
        if not any(client_info['counselor_id'] == room for client_info in clients.values()):
            print(f"No more clients in room {room}, stopping data transmission")
            if room in threads:
                running_threads[room] = False
                threads[room].join()
                del threads[room]
                del running_threads[room]

@app.route('/')
def index():
    return "Hello World!"






def start_flask():
    socketio.run(app, host='0.0.0.0', port=5000)



# WebSocket 서버 시작
async def main():
    # 두 WebSocket 서버를 설정하고 실행
    server1 = await websockets.serve(handle_connection, "localhost", 8765)
    server2 = await websockets.serve(full_handle, "localhost", 8766)

    print("Starting both WebSocket servers...")

    # Flask-SocketIO 서버를 별도로 실행
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, start_flask)

    # 서버가 계속 실행되도록 대기
    await asyncio.Future()  # 무한 대기를 통해 서버 유지

if __name__ == "__main__":
    asyncio.run(main())