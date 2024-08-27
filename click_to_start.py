from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")


connected_clients = 0


@socketio.on('start')
def handle_start():
    print("start")
    socketio.emit('sensor_start')


@socketio.on('connect')
def handle_connect():
    global connected_clients
    connected_clients += 1
    print('Client connected')
    check_clients()

@socketio.on('disconnect')
def handle_disconnect():
    global connected_clients
    connected_clients -= 1
    print('Client disconnected')
    check_clients()

def check_clients():
    if connected_clients == 2:
        socketio.emit('enable_start', {'enable': True}, to=None)
        print("ttttt")
    else:
        socketio.emit('enable_start', {'enable': False}, to=None)

@app.route('/')
def index():
    return "Hello World!"

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)