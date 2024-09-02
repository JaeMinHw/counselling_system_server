from flask import Flask, request
from flask_socketio import SocketIO, emit
import threading
import random
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

clients = {}  # 클라이언트별 상태를 추적하기 위한 딕셔너리
threads = {}  # 클라이언트별 스레드를 추적하기 위한 딕셔너리

# 임의의 데이터를 주기적으로 전송하는 함수
def generate_data(client_id):
    while clients.get(client_id, {}).get('active', False):
        data1 = random.uniform(0, 100)
        data2 = random.uniform(0, 100)
        socketio.emit('data_update', {'data1': data1, 'data2': data2}, room=clients[client_id]['session_id'])
        # socketio.emit('data_update', {'data1': data1, 'data2': data2})
        # print(f"Sent data to {client_id}: data1={data1}, data2={data2}")  # 디버깅을 위한 출력
        time.sleep(0)
    print(f"Thread for client {client_id} has stopped.")  # 스레드 종료 로그

def start_background_thread(client_id):
    # 스레드를 시작하고 딕셔너리에 저장
    thread = threading.Thread(target=generate_data, args=(client_id,))
    threads[client_id] = thread
    thread.start()


@socketio.on('start')
def handle_start():
    counselor_id = [k for k, v in clients.items() if v['session_id'] == request.sid][0]
    patient_id = clients[counselor_id]['pair']
    
    print(f"Start data transmission for patient {patient_id} by counselor {counselor_id}")
    clients[patient_id]['active'] = True  # 환자 클라이언트 상태를 활성화
    socketio.emit('sensor_start', room=clients[patient_id]['session_id'])  # 시작 신호를 환자 클라이언트로 전송
    print("tetetete",patient_id)
    start_background_thread(patient_id)  # 데이터 생성 시작


@socketio.on('patient_login')
def handle_patient_login(data):
    session_id = request.sid
    patient_id = data['patient_id']
    counselor_id = data['counselor_id']
    
    clients[patient_id] = {
        'type': 'patient',
        'session_id': session_id,
        'counselor_id': counselor_id,
        'active': False
    }
    
    print(f"Patient {patient_id} logged in with counselor {counselor_id}")

@socketio.on('counselor_login')
def handle_counselor_login(data):
    session_id = request.sid
    counselor_id = data['counselor_id']

    print(counselor_id) # counselor_1
    
    clients[counselor_id] = {
        'type': 'counselor',
        'session_id': session_id,
        'active': False
    }
    
    # 대기 중인 환자가 있는지 확인
    for patient_id, patient_info in clients.items():
        if patient_info['type'] == 'patient' and patient_info['counselor_id'] == counselor_id and not patient_info['active']:
            clients[counselor_id]['pending_connection'] = patient_id
            socketio.emit('connection_request', {'message': f'Connect with patient {patient_id}?'}, room=session_id)
            print(f"Counselor {counselor_id} prompted to connect with patient {patient_id}")
            break

@socketio.on('accept_connection')
def handle_accept_connection():
    counselor_id = [k for k, v in clients.items() if v['session_id'] == request.sid][0]
    patient_id = clients[counselor_id].pop('pending_connection', None)
    
    if patient_id:
        # 클라이언트 상태 활성화
        clients[counselor_id]['active'] = True
        clients[patient_id]['active'] = True
        
        # 두 클라이언트 연결
        clients[counselor_id]['pair'] = patient_id
        clients[patient_id]['pair'] = counselor_id
        
        # 연결된 클라이언트들에게 연결 성공 알림
        socketio.emit('connection_accepted', {'message': f'Connected with counselor {counselor_id}'}, room=clients[patient_id]['session_id'])
        socketio.emit('connection_accepted', {'message': f'Connected with patient {patient_id}'}, room=clients[counselor_id]['session_id'])

        print(f"Counselor {counselor_id} connected with patient {patient_id}")


@socketio.on('stop')
def handle_stop():
    counselor_id = [k for k, v in clients.items() if v['session_id'] == request.sid][0]
    patient_id = clients[counselor_id]['pair']
    
    print(f"Stop data transmission for patient {patient_id} by counselor {counselor_id}")
    clients[patient_id]['active'] = False  # 환자 클라이언트 상태를 비활성화
    
    # 스레드 종료 처리
    if patient_id in threads:
        threads[patient_id].join()  # 스레드가 종료될 때까지 기다림
        del threads[patient_id]  # 스레드 종료 후 딕셔너리에서 제거

@socketio.on('decline_connection')
def handle_decline_connection():
    counselor_id = [k for k, v in clients.items() if v['session_id'] == request.sid][0]
    pending_patient = clients[counselor_id].pop('pending_connection', None)
    print(f"Counselor {counselor_id} declined connection request from patient {pending_patient}")

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    client_id = [k for k, v in clients.items() if v['session_id'] == session_id][0]
    client_info = clients.pop(client_id, None)
    
    if client_info and client_info['active']:
        paired_id = client_info.get('pair')
        if paired_id:
            clients[paired_id]['active'] = False
            socketio.emit('disconnected', {'message': 'Your pair has disconnected.'}, room=clients[paired_id]['session_id'])
            print(f"Client {client_id} and its pair {paired_id} disconnected")
    
    # 스레드가 종료될 때까지 기다림
    if client_id in threads:
        threads[client_id].join()
        del threads[client_id]  # 스레드 종료 후 딕셔너리에서 제거

@app.route('/')
def index():
    return "Hello World!"

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
