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
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
import threading
import random
import time
import pymysql
from datetime import datetime, timedelta
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# --------------------------------------------
# (1) Whisper 라이브러리 추가
import whisper
import soundfile as sf   # audio_wav -> numpy array 변환 용도
# --------------------------------------------

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
from flask_cors import CORS

CORS(app, support_credentials=True, origins='*')
clients = {}
threads = {}
running_threads = {}

# --------------------------------------------
# (2) Google Speech 관련 부분 제거/주석 처리
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "c:/Users/User/stone-climate-441309-e7-69ab160b40d4.json"
# from google.cloud import speech
# speech_client = speech.SpeechClient()
# --------------------------------------------

# (3) Whisper STT 모델 로드
stt_model = whisper.load_model("medium") 
# "tiny", "base", "small", "medium", "large" 등 다양한 모델이 있음

# 모델 및 스케일러 로드 (화자 분류 모델)
model = tf.keras.models.load_model("cnn_lstm_autoencoder_me3.h5")
scaler = joblib.load("scaler_me3.pkl")
threshold = 0.083

output_folder = "received_full_audio"
os.makedirs(output_folder, exist_ok=True)



def get_db_connection():
    return pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='1234',
        database='counselling_sys',
        cursorclass=pymysql.cursors.DictCursor
    )








# 음성 특징 추출 함수 (화자 판별용)
def extract_mfcc(audio_data):
    y, sr = librosa.load(io.BytesIO(audio_data), sr=16000)
    y = y.astype(np.float32)  # float32 변환
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=1024)
    return np.mean(mfccs.T, axis=0).astype(np.float32)  # float32 유지

# 화자 예측 함수 (CNN-LSTM Autoencoder)
def predict_voice(audio_data):
    features = extract_mfcc(audio_data).reshape(1, -1).astype(np.float32)  # float32 변환
    features = scaler.transform(features).astype(np.float32)  # 스케일러 적용 후 변환

    reconstructed = model.predict(features.astype(np.float32))  # 모델 입력도 float32 유지
    reconstructed = reconstructed.astype(np.float32)  # 모델 출력도 float32로 변환

    mse = np.mean((features - reconstructed) ** 2, dtype=np.float32)  # MSE 계산도 float32 유지
    print(f"MSE: {mse}")

    label = "me" if mse < threshold else "another"
    return label, float(mse)

# Whisper 기반 STT 함수
def transcribe_audio(audio_wav):
    audio_wav.seek(0)
    audio_content = audio_wav.read()
    if not audio_content or len(audio_content) == 0:
        raise ValueError("Audio data is empty or invalid.")

    # soundfile을 이용해 BytesIO -> PCM 로드
    audio_wav.seek(0)
    samples, sr = sf.read(io.BytesIO(audio_content), dtype="float32")  # float32 변환

    # Whisper 모델에 전달
    result = stt_model.transcribe(samples, language="ko", fp16=False)
    text = result.get("text", "")

    return text



# VAD로 음성 구간 탐지
def detect_voice_segments(audio_data):
    vad = webrtcvad.Vad()
    vad.set_mode(0)  # 가장 민감한 모드

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
def process_audio(audio_data, sent_time):
    voice_segments = detect_voice_segments(audio_data)  # 음성 구간 탐지
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
    audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)

    total_duration = len(audio) / 1000  # 전체 오디오 길이 (초 단위)
    sent_time_dt = datetime.strptime(sent_time, "%H:%M:%S")  # sentTime을 datetime으로 변환

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

                # 화자 판별(스피커 구분)
                label, mse = predict_voice(audio_wav.read())

                # 다시 포인터 맨 앞으로
                audio_wav.seek(0)

                # Whisper로 텍스트 변환
                text = transcribe_audio(audio_wav)

                 # 절대 시간 계산
                segment_start_time = sent_time_dt - timedelta(seconds=total_duration) + timedelta(seconds=segment_start)
                segment_end_time = segment_start_time + timedelta(seconds=(segment_end - segment_start))

                # MIN_TEXT_LENGTH = 3  # 한글 글자 기준
                # # 텍스트가 존재할 경우에만 파일 저장
                # if len(text.strip()) >= MIN_TEXT_LENGTH:
                #     timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                #     filename = f"text_detected_{label}_{timestamp}.wav"
                #     save_path = os.path.join("detected_speech_audio", filename)
                #     os.makedirs("detected_speech_audio", exist_ok=True)
                    
                #     with open(save_path, "wb") as f:
                #         f.write(audio_wav.getbuffer())
                #     print(f"[SAVED] Speech detected and saved to {save_path}")


               

                # text = "안녕하세요안녕하세요안녕하세요안녕하세요안녕하세요안녕하세요"

                
                connection = get_db_connection()

                try :
                    with connection.cursor() as cursor:
                        sql = "insert into conversation_data(start, label, mse, text, counseling_id)"
                        cursor.execute(sql, (segment_start_time.strftime("%H:%M:%S"), label, mse, text,  ))
                        pass
                finally:
                    connection.close()


                results.append({
                    "label": label,      # 누구인지(me or another)
                    "mse": mse,         # MSE 추가
                    "text": text,       # 실제 Whisper 인식 결과
                    "start": segment_start_time.strftime("%H:%M:%S"),
                    "send_time": segment_start_time.strftime("%H:%M:%S"),
                    "end_time": segment_end_time.strftime("%H:%M:%S"),
                })

                current_segment = []

    return results

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


# ----------------------------------------------
# 이하 DB/Flask-SocketIO 관련 기존 코드 그대로
# ----------------------------------------------




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
        socketio.emit('data_update', {'data1': data1, 'data2': data2, 'data3': data3, 'time': current_time}, room=room)
        time.sleep(0.1)

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


# 여기서 counseling_data insert 하기
@socketio.on('accept_connection')
def handle_accept_connection(data):
    # client_id = data['client']
    counselor_id = clients[request.sid]['counselor_id']
    print("test")
    print(data)
    
    room = counselor_id
    
    socketio.emit('connection_accepted', {'message': 'Connection accepted'}, room=room)
    print(f"Started data transmission for room {room}")

    connection = get_db_connection()

    try :
        with connection.cursor() as cursor:#counseling_id 는 알아서 증가하거나 그래야 하고 clients_id는 data 넣어주면 되고
            sql = "insert into counseling_data (clients_id) values(%s)"
            cursor.execute(sql, (data,))

            pass
    finally:
        connection.close()



@socketio.on('start')
def handle_start():
    counsel_id = clients[request.sid]['counselor_id']
    room = counsel_id
    socketio.emit('sensor_start', room)

@socketio.on('sensor_data_batch')
def sensor_data_batch(data):
    room = data['room']
    data1_batch = data['data1_batch']
    current_time = data['time']

    socketio.emit('data_update_batch', {
        'data1_batch': data1_batch,
        'time': current_time
    }, room=room)

@socketio.on('sensor_data')
def sensor_data(data):
    room = data['room']
    sensor = data['sensor']
    value = data['value']
    current_time = data['time']

    if sensor == 'data2':
        socketio.emit('data_update', {
            'sensor': 'data2',
            'value': value,
            'time': current_time
        }, room=room)

    elif sensor == 'data3':
        socketio.emit('data_update', {
            'sensor': 'data3',
            'value': value,
            'time': current_time
        }, room=room)

@socketio.on('stop')
def handle_stop():
    counsel_id = clients[request.sid]['counselor_id']
    room = counsel_id
    socketio.emit('stop', room)
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

def feture_detect(data):
    socketio.emit('feature_detect', data)

@app.route("/feature")
def feture_detect_test():
    now = time
    data = {
        'time': now.strftime('%H:%M:%S'),
        'feature': 'anxiety'
    }
    feture_detect(data)
    return data



@app.route('/clients/<string:counselling_id>')
def client(counselling_id):
    connection = get_db_connection()
    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = "SELECT username,id FROM clients WHERE counsel_id = %s"
            cursor.execute(sql, (counselling_id,))
            result = cursor.fetchall()
            print(result)
            return jsonify(result)
    finally:
        connection.close()


@app.route('/counsel_name/<string:counsel_id>')
def counsel_id(counsel_id):
    connection = get_db_connection()
    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = "SELECT counsel_name FROM counsel WHERE counsel_id = %s"
            cursor.execute(sql, (counsel_id,))
            result = cursor.fetchone()
            print(result)
            return result
    finally:
        connection.close()




@app.route('/')
def index():
    return "Hello World!"


def start_flask():
    socketio.run(app, host='0.0.0.0', port=5000)

async def main():
    server1 = await websockets.serve(handle_connection, "localhost", 8765, max_size=None)
    server2 = await websockets.serve(full_handle, "localhost", 8766, max_size=None)

    print("Starting both WebSocket servers...")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, start_flask)

    # 서버가 계속 실행되도록 대기
    await asyncio.Future()  # 무한 대기를 통해 서버 유지

if __name__ == "__main__":
    asyncio.run(main())
