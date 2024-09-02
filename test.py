from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room
import threading
import random
import time
import pymysql

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

clients = {}  # 클라이언트별 상태를 추적하기 위한 딕셔너리
threads = {}  # 클라이언트별 스레드를 추적하기 위한 딕셔너리

# 데이터베이스 연결 함수
def get_db_connection():
    return pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='1234',
        database='counselling_sys',
        cursorclass=pymysql.cursors.DictCursor
    )

# 데이터베이스에서 counselor_id 조회
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

# 데이터베이스에서 patient_id 조회 (예: counselor_login 시 활용 가능)
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

# 임의의 데이터를 주기적으로 전송하는 함수
def generate_data(room):
    while True:
        data1 = random.uniform(0, 100)
        data2 = random.uniform(0, 100)
        print(f"Sending data to room {room}")
        socketio.emit('data_update', {'data1': data1, 'data2': data2}, room=room)
        time.sleep(0)

def start_background_thread(room):
    thread = threading.Thread(target=generate_data, args=(room,))
    thread.start()
    return thread

@socketio.on('patient_login')
def handle_patient_login(data):
    patient_id = data['patient_id']
    counselor_id = get_counselor_id_by_patient(patient_id)  # DB에서 counselor_id 조회
    
    if counselor_id is None:
        print(f"Could not find counselor_id for patient {patient_id}")
        return
    
    session_id = request.sid  # 현재 연결된 클라이언트의 세션 ID
    clients[session_id] = {
        'type': 'patient',
        'patient_id': patient_id,
        'counselor_id': counselor_id
    }
    
    room = counselor_id  # 상담사 ID를 기반으로 room을 설정
    join_room(room)
    print(f"Patient {patient_id} joined room {room}")
    
    # 해당 상담사가 연결되어 있는지 확인
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
    
    room = counselor_id  # 상담사 ID를 기반으로 room을 설정
    join_room(room)
    print(f"Counselor {counselor_id} joined room {room}")
    
    # 데이터베이스에서 이 상담사와 연결된 모든 환자를 조회
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


@socketio.on('start')
def handle_start():
    counsel_id = clients[request.sid]['counselor_id']
    room = counsel_id
    print(f"Start data transmission for room {room}")
    
    if room in threads:
        print(f"Data transmission already running for room {room}")
    else:
        start_background_thread(room)

@socketio.on('stop')
def handle_stop():
    counsel_id = clients[request.sid]['counselor_id']
    room = counsel_id
    
    print(f"Stop data transmission for room {room}")
    if room in threads:
        threads[room].join()
        del threads[room]
        print(f"Data transmission stopped for room {room}")

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    client_info = clients.pop(session_id, None)
    
    if client_info:
        room = client_info['counselor_id']
        leave_room(room)
        print(f"Client {session_id} left room {room}")
        
        # 방에 남은 클라이언트가 있는지 확인
        if not any(client_info['counselor_id'] == room for client_info in clients.values()):
            print(f"No more clients in room {room}, stopping data transmission")
            if room in threads:
                threads[room].join()
                del threads[room]

@app.route('/')
def index():
    return "Hello World!"

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
