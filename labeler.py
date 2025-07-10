from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

import json


from db import CLASSIFICATION_OPTIONS, AutoFEDDatabase


app = Flask(__name__)
socketio = SocketIO(app)

connected_clients = {}    


@app.route('/')
def index():
    """Show a list of available runs"""
    with AutoFEDDatabase() as conn:
        runs = conn.available_runs()

    return render_template('index.html', runs=runs)


@app.route('/label/<run_name>')
def show_videos(run_name):
    """Show videos for a specific run"""

    with AutoFEDDatabase() as conn:
        videos = conn.all_videos_for_run("video_labels", run_name)
    
    if not videos:
        return render_template('error.html', message=f"No videos found for run: {run_name}"), 404
    
    return render_template('videos.html', 
                          videos=videos, 
                          run_name=run_name,
                          classification_options=CLASSIFICATION_OPTIONS)


@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)
    connected_clients[request.sid] = request


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)

    if request.sid in connected_clients:
        del connected_clients[request.sid]


@socketio.on('update-video-classifications')
def set_classification_state(data):
    id = data.get('id')
    classifications = json.dumps(data.get('classifications'))

    try:
        with AutoFEDDatabase() as conn:
            conn.set_classifications(id, classifications)

        for sid in connected_clients:
            if sid != request.sid:
                socketio.emit('checkbox-update', {'id': id, 'classifications': json.loads(classifications)}, room=sid)

        return jsonify({'message': 'Video classifications updated successfully'}), 200
    except Exception as e:
        print(f"Error: {e}")

        return jsonify({'message': f'Error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
