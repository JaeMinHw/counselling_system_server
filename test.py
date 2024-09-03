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
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Sending data to room {room}")
        socketio.emit('data_update', {'data1': data1, 'data2': data2, 'time': current_time}, room=room)
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

@socketio.on('start')
def handle_start():
    counsel_id = clients[request.sid]['counselor_id']
    room = counsel_id
    print(f"Start data transmission for room {room}")
    
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

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
