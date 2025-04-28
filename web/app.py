from flask import Flask, render_template, Response, jsonify, request, make_response, send_file, send_from_directory
from flask_socketio import SocketIO
import cv2
import numpy as np
import os
import json
from datetime import datetime
import threading
import time
import sys
import base64
from io import BytesIO
from PIL import Image
import shutil
from settings_manager import SettingsManager

# Add the parent directory to the path so we can import the security module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from security.face_detection import detection
from security.config import (
    UPLOADS_DIR, SNAPSHOTS_MOTION_DIR, SNAPSHOTS_UNKNOWN_DIR, WEB_DIR
)
from database import (
    init_db, get_db, get_statistics, get_recent_events, get_known_faces,
    get_snapshots, get_email_logs, clear_events, KnownFace, UnknownFace,
    MotionEvent, EmailLog
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'security_dashboard_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize database and settings
init_db()
settings_manager = SettingsManager()

# Global variables
detector = None
camera_thread = None
camera_running = False
camera_lock = threading.Lock()
latest_frame = None
frame_lock = threading.Lock()
camera_cap = None
last_error_log_time = {
    'camera_not_running': 0,
    'no_frame': 0,
    'invalid_dimensions': 0,
    'encoding_error': 0,
    'general_error': 0
}
error_log_interval = 5  # Log errors at most once every 5 seconds

# Configuration
config = settings_manager.settings

def get_statistics():
    """Get statistics from database"""
    db = next(get_db())
    stats = {
        'known_faces': db.query(KnownFace).count(),
        'unknown_faces': db.query(UnknownFace).count(),
        'motion_events': db.query(MotionEvent).count(),
        'email_notifications': db.query(EmailLog).count()
    }
    return stats

def get_recent_events(limit=10):
    """Get recent events from database"""
    db = next(get_db())
    events = []
    
    # Get known faces
    known_faces = db.query(KnownFace).order_by(KnownFace.id.desc()).limit(limit).all()
    for face in known_faces:
        events.append({
            'type': 'known_face',
            'name': face.name,
            'date': face.date,
            'time': face.time,
            'confidence': f"{face.confidence:.2f}",
            'snapshot': face.snapshot,
            'status': face.status
        })
    
    # Get unknown faces
    unknown_faces = db.query(UnknownFace).order_by(UnknownFace.id.desc()).limit(limit).all()
    for face in unknown_faces:
        events.append({
            'type': 'unknown_face',
            'name': 'Unknown Person',
            'date': face.date,
            'time': face.time,
            'confidence': f"{face.confidence:.2f}",
            'snapshot': face.snapshot,
            'status': face.status
        })
    
    # Get motion events
    motion_events = db.query(MotionEvent).order_by(MotionEvent.id.desc()).limit(limit).all()
    for event in motion_events:
        events.append({
            'type': 'motion',
            'name': 'Motion Event',
            'date': event.date,
            'time': event.time,
            'confidence': f"{event.confidence:.2f}",
            'snapshot': event.snapshot,
            'status': event.status
        })
    
    # Sort all events by date and time
    events.sort(key=lambda x: (x['date'], x['time']), reverse=True)
    return events[:limit]

def get_known_faces():
    """Get list of known faces from the uploads/images directory"""
    known_faces = []
    try:
        for filename in os.listdir(UPLOADS_DIR):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                known_faces.append({
                    'name': name,
                    'filename': filename
                })
    except Exception as e:
        print(f"Error reading known faces: {str(e)}")
    
    return known_faces

def get_snapshots(limit=20):
    """Get list of snapshots from the snapshots directories"""
    snapshots = []
    
    # Get motion snapshots
    try:
        print(f"Checking motion snapshots directory: {SNAPSHOTS_MOTION_DIR}")
        if not os.path.exists(SNAPSHOTS_MOTION_DIR):
            print(f"Motion snapshots directory does not exist: {SNAPSHOTS_MOTION_DIR}")
        else:
            motion_files = os.listdir(SNAPSHOTS_MOTION_DIR)
            print(f"Found {len(motion_files)} files in motion directory")
            for filename in motion_files:
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(SNAPSHOTS_MOTION_DIR, filename)
                    file_stat = os.stat(filepath)
                    snapshots.append({
                        'type': 'motion',
                        'filename': filename,
                        'filepath': f'/snapshot/motion/{filename}',
                        'date': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    })
    except Exception as e:
        print(f"Error reading motion snapshots: {str(e)}")
    
    # Get unknown face snapshots
    try:
        print(f"Checking unknown face snapshots directory: {SNAPSHOTS_UNKNOWN_DIR}")
        if not os.path.exists(SNAPSHOTS_UNKNOWN_DIR):
            print(f"Unknown face snapshots directory does not exist: {SNAPSHOTS_UNKNOWN_DIR}")
        else:
            unknown_files = os.listdir(SNAPSHOTS_UNKNOWN_DIR)
            print(f"Found {len(unknown_files)} files in unknown directory")
            for filename in unknown_files:
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(SNAPSHOTS_UNKNOWN_DIR, filename)
                    file_stat = os.stat(filepath)
                    snapshots.append({
                        'type': 'unknown_face',
                        'filename': filename,
                        'filepath': f'/snapshot/unknown/{filename}',
                        'date': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    })
    except Exception as e:
        print(f"Error reading unknown face snapshots: {str(e)}")
    
    # Sort snapshots by date (most recent first)
    snapshots.sort(key=lambda x: x['date'], reverse=True)
    
    # Return only the most recent snapshots
    return snapshots[:limit]

def get_email_logs(limit=50, include_confidence=False):
    """Get email logs from database"""
    db = next(get_db())
    email_logs = db.query(EmailLog).order_by(EmailLog.id.desc()).limit(limit).all()
    return [{
        'type': 'email',
        'date': log.date,
        'time': log.time,
        'recipient': log.recipient,
        'subject': log.subject,
        'status': log.status
    } for log in email_logs]

def camera_stream():
    """Camera stream thread function"""
    global camera, latest_frame, detector, camera_running
    
    try:
        # Initialize camera with retries
        max_retries = 5
        retry_count = 0
        while retry_count < max_retries and camera_running:
            try:
                # Try different camera backends
                for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                    try:
                        camera = cv2.VideoCapture(0, backend)
                        if camera.isOpened():
                            # Only log successful initialization once
                            if not hasattr(app, 'camera_init_logged'):
                                print(f"Camera initialized successfully with backend {backend}")
                                app.camera_init_logged = True
                            break
                    except Exception as e:
                        # Only log backend errors once
                        if not hasattr(app, f'backend_error_{backend}_logged'):
                            print(f"Failed to initialize camera with backend {backend}: {str(e)}")
                            setattr(app, f'backend_error_{backend}_logged', True)
                        continue
                
                if camera.isOpened():
                    break
                
                retry_count += 1
                if retry_count < max_retries:
                    # Only log retry attempts occasionally
                    if not hasattr(app, 'last_retry_log_time') or time.time() - app.last_retry_log_time > 5:
                        app.last_retry_log_time = time.time()
                        print(f"Retrying camera initialization ({retry_count}/{max_retries})")
                    time.sleep(1)
            except Exception as e:
                # Only log initialization errors occasionally
                if not hasattr(app, 'last_init_error_time') or time.time() - app.last_init_error_time > 5:
                    app.last_init_error_time = time.time()
                    print(f"Error initializing camera: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(1)
        
        if not camera.isOpened():
            print("Failed to initialize camera after maximum retries")
            camera_running = False
            return
        
        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config['frame_width'])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config['frame_height'])
        camera.set(cv2.CAP_PROP_FPS, config['fps'])
        
        # Initialize blank frame
        blank_frame = np.zeros((config['frame_height'], config['frame_width'], 3), dtype=np.uint8)
        latest_frame = blank_frame.copy()
        
        # Initialize detector if not already done
        if detector is None:
            detector = detection()
            # Set email notifications based on config
            if detector.email_notifier:
                if config['email_enabled']:
                    detector.email_notifier.enable_emails()
                else:
                    detector.email_notifier.disable_emails()
                # Only log email status once
                if not hasattr(app, 'email_status_logged'):
                    print(f"Email notifications {'enabled' if config['email_enabled'] else 'disabled'}")
                    app.email_status_logged = True
        
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while camera_running:
            try:
                ret, frame = camera.read()
                if ret and frame is not None:
                    # Process frame for detection
                    if detector:
                        processed_frame = detector.process_frame(frame)
                        if processed_frame is not None:
                            # Update latest frame with the processed frame that includes detections
                            latest_frame = processed_frame.copy()
                        else:
                            # If no processed frame is returned, use the original frame
                            latest_frame = frame.copy()
                    else:
                        latest_frame = frame.copy()
                    
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    # Only log frame read failures occasionally
                    if consecutive_failures % 5 == 0:
                        print(f"Failed to read frame ({consecutive_failures}/{max_consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        print("Too many consecutive failures, attempting to reinitialize camera")
                        camera.release()
                        time.sleep(1)
                        camera = cv2.VideoCapture(0)
                        if not camera.isOpened():
                            print("Failed to reinitialize camera")
                            camera_running = False
                            break
                        consecutive_failures = 0
                
                # Reduce sleep time to minimize lag
                time.sleep(0.01)
                
            except Exception as e:
                # Only log stream errors occasionally
                if not hasattr(app, 'last_stream_error_time') or time.time() - app.last_stream_error_time > 5:
                    app.last_stream_error_time = time.time()
                    print(f"Error in camera stream: {str(e)}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print("Too many consecutive failures, stopping camera")
                    camera_running = False
                    break
                time.sleep(1)
    
    except Exception as e:
        # Only log fatal errors once
        if not hasattr(app, 'fatal_error_logged'):
            print(f"Fatal error in camera stream: {str(e)}")
            app.fatal_error_logged = True
        camera_running = False
    
    finally:
        if camera is not None:
            camera.release()
        camera_running = False
        # Only log camera stop once
        if not hasattr(app, 'camera_stop_logged'):
            print("Camera stream stopped")
            app.camera_stop_logged = True

def start_camera():
    """Start the camera in a separate thread"""
    global camera_thread, camera_running
    
    with camera_lock:
        if not camera_running:
            camera_running = True
            camera_thread = threading.Thread(target=camera_stream)
            camera_thread.daemon = True
            camera_thread.start()
            return True
        return False

def stop_camera():
    """Stop the camera thread"""
    global camera_running, camera, detector, latest_frame
    
    with camera_lock:
        if camera_running:
            print("Stopping camera...")
            camera_running = False
            
            # Release camera resources
            if camera is not None:
                camera.release()
                camera = None
            
            # Wait for thread to finish
            if camera_thread and camera_thread.is_alive():
                camera_thread.join(timeout=2.0)
            
            # Reset detector to ensure clean state
            if detector:
                detector.reset()
            
            # Clear latest frame
            latest_frame = None
            
            print("Camera stopped successfully")
            return True
        return False

@app.route('/')
def index():
    """Render the main dashboard page"""
    stats = get_statistics()
    recent_events = get_recent_events(5)
    recent_emails = get_email_logs(5, include_confidence=False)  # Get last 5 emails without confidence
    return render_template('index.html', stats=stats, recent_events=recent_events, recent_emails=recent_emails)

@app.route('/camera')
def camera():
    """Render the camera page"""
    return render_template('camera.html')

@app.route('/events')
def events():
    """Render the events page"""
    events = get_recent_events(50)
    return render_template('events.html', events=events)

@app.route('/snapshots')
def snapshots():
    """Render the snapshots page"""
    snapshots = get_snapshots(50)
    return render_template('snapshots.html', snapshots=snapshots)

@app.route('/faces')
def faces():
    """Render the faces management page"""
    known_faces = get_known_faces()
    return render_template('faces.html', known_faces=known_faces)

@app.route('/settings')
def settings():
    """Render the settings page"""
    return render_template('settings.html', config=config)

@app.route('/api/statistics')
def api_statistics():
    stats = get_statistics()
    return jsonify(stats)

@app.route('/api/events')
def api_events():
    db = next(get_db())
    events = get_recent_events(db)
    return jsonify(events)

@app.route('/api/snapshots')
def api_snapshots():
    """API endpoint to get snapshots"""
    try:
        snapshots = []
        
        # Get snapshots from motion events
        if os.path.exists(SNAPSHOTS_MOTION_DIR):
            for filename in os.listdir(SNAPSHOTS_MOTION_DIR):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(SNAPSHOTS_MOTION_DIR, filename)
                    file_stat = os.stat(filepath)
                    snapshots.append({
                        'type': 'motion',
                        'filename': filename,
                        'filepath': f'/snapshot/motion/{filename}',
                        'date': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    })
        
        # Get snapshots from unknown faces
        if os.path.exists(SNAPSHOTS_UNKNOWN_DIR):
            for filename in os.listdir(SNAPSHOTS_UNKNOWN_DIR):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(SNAPSHOTS_UNKNOWN_DIR, filename)
                    file_stat = os.stat(filepath)
                    snapshots.append({
                        'type': 'unknown_face',
                        'filename': filename,
                        'filepath': f'/snapshot/unknown/{filename}',
                        'date': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    })
        
        # Sort snapshots by date (most recent first)
        snapshots.sort(key=lambda x: x['date'], reverse=True)
        
        return jsonify(snapshots)
    except Exception as e:
        print(f"Error getting snapshots: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/snapshot/<path:filename>')
def serve_snapshot(filename):
    """Serve snapshot images"""
    try:
        print(f"Attempting to serve snapshot: {filename}")
        
        # Determine which directory to serve from based on the URL path
        if filename.startswith('motion/'):
            # Try both directory structures
            directories = [
                SNAPSHOTS_MOTION_DIR,  # New structure
                os.path.join(WEB_DIR, 'snapshots', 'motion')  # Old structure
            ]
            filename = filename[6:].lstrip('/')  # Remove 'motion/' prefix and any leading slashes
            print(f"Motion snapshot requested. Filename: {filename}")
        elif filename.startswith('unknown/'):
            # Try both directory structures
            directories = [
                SNAPSHOTS_UNKNOWN_DIR,  # New structure
                os.path.join(WEB_DIR, 'snapshots', 'unknown_faces')  # Old structure
            ]
            filename = filename[8:].lstrip('/')  # Remove 'unknown/' prefix and any leading slashes
            print(f"Unknown face snapshot requested. Filename: {filename}")
        else:
            print(f"Invalid snapshot path: {filename}")
            return "Invalid snapshot path", 400

        # Try each directory until we find the file
        for directory in directories:
            filepath = os.path.join(directory, filename)
            print(f"Trying filepath: {filepath}")
            print(f"Directory exists: {os.path.exists(directory)}")
            print(f"File exists: {os.path.exists(filepath)}")
            
            if os.path.exists(filepath):
                # Serve the file
                return send_file(filepath, mimetype='image/jpeg')
            
            # If directory exists but file doesn't, list its contents
            if os.path.exists(directory):
                print(f"Contents of {directory}:")
                for f in os.listdir(directory):
                    print(f"  - {f}")

        # If we get here, the file wasn't found in any directory
        return "File not found", 404
    except Exception as e:
        print(f"Error serving snapshot: {str(e)}")
        return "Error serving snapshot", 500

@app.route('/face/<path:filename>')
def serve_face_image(filename):
    """Serve face images from the uploads directory"""
    try:
        # Get the absolute path to the uploads directory
        uploads_dir = os.path.abspath(UPLOADS_DIR)
        filepath = os.path.join(uploads_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Face image not found: {filepath}")
            return "Image not found", 404
            
        return send_file(filepath, mimetype='image/jpeg')
    except Exception as e:
        print(f"Error serving face image: {str(e)}")
        return "Error serving image", 500

@app.route('/api/faces')
def api_faces():
    db = next(get_db())
    faces = get_known_faces(db)
    return jsonify([{
        'name': face.name,
        'date': face.date,
        'time': face.time,
        'confidence': face.confidence,
        'snapshot': face.snapshot,
        'status': face.status
    } for face in faces])

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    """API endpoint to get or update settings"""
    global config
    
    if request.method == 'GET':
        return jsonify(config)
    
    elif request.method == 'POST':
        data = request.json
        
        # Update config with all provided settings
        for key, value in data.items():
            if key in config:
                # Convert numeric values to appropriate type
                if key in ['max_snapshots', 'max_events', 'frame_width', 'frame_height', 'fps', 'smtp_port']:
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        continue
                # Convert boolean values
                elif key in ['email_enabled', 'auto_cleanup']:
                    value = bool(value)
                config[key] = value
        
        # Handle email notifier if email settings changed
        if 'email_enabled' in data and detector and detector.email_notifier:
            if config['email_enabled']:
                detector.email_notifier.enable_emails()
            else:
                detector.email_notifier.disable_emails()
            print(f"Email notifications {'enabled' if config['email_enabled'] else 'disabled'}")
        
        # Save settings to file
        settings_manager.update_settings(config)
        
        return jsonify({'status': 'success', 'config': config})

@app.route('/api/settings/email', methods=['POST'])
def api_settings_email():
    """API endpoint to update email settings"""
    global config
    
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['email_enabled', 'smtp_server', 'smtp_port', 'smtp_username', 
                         'smtp_password', 'sender_email', 'recipient_email']
        for field in required_fields:
            if field not in data:
                return jsonify({'status': 'error', 'message': f'Missing required field: {field}'}), 400
        
        # Update email settings with validation
        config['email_enabled'] = bool(data['email_enabled'])
        
        # Validate SMTP server
        if not data['smtp_server']:
            return jsonify({'status': 'error', 'message': 'SMTP server cannot be empty'}), 400
        config['smtp_server'] = data['smtp_server']
        
        # Validate SMTP port
        try:
            port = int(data['smtp_port'])
            if port < 1 or port > 65535:
                return jsonify({'status': 'error', 'message': 'Invalid SMTP port'}), 400
            config['smtp_port'] = port
        except ValueError:
            return jsonify({'status': 'error', 'message': 'Invalid SMTP port'}), 400
        
        # Validate email addresses
        if not data['sender_email'] or '@' not in data['sender_email']:
            return jsonify({'status': 'error', 'message': 'Invalid sender email'}), 400
        config['sender_email'] = data['sender_email']
        
        if not data['recipient_email'] or '@' not in data['recipient_email']:
            return jsonify({'status': 'error', 'message': 'Invalid recipient email'}), 400
        config['recipient_email'] = data['recipient_email']
        
        # Update username and password
        config['smtp_username'] = data['smtp_username']
        config['smtp_password'] = data['smtp_password']
        
        # Update email notifier if it exists
        if detector and detector.email_notifier:
            if config['email_enabled']:
                detector.email_notifier.enable_emails()
            else:
                detector.email_notifier.disable_emails()
            
            # Update email notifier configuration
            detector.email_notifier.config.update({
                'smtp_server': config['smtp_server'],
                'smtp_port': config['smtp_port'],
                'sender_email': config['sender_email'],
                'sender_password': config['smtp_password'],
                'receiver_email': config['recipient_email']
            })
            
            print(f"Email notifications {'enabled' if config['email_enabled'] else 'disabled'}")
            print(f"SMTP Server: {config['smtp_server']}:{config['smtp_port']}")
            print(f"Sender: {config['sender_email']}")
            print(f"Recipient: {config['recipient_email']}")
        
        # Save settings to file
        settings_manager.update_settings(config)
        
        return jsonify({'status': 'success', 'config': config})
        
    except Exception as e:
        print(f"Error updating email settings: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/settings/storage', methods=['POST'])
def api_settings_storage():
    """API endpoint to update storage settings"""
    global config
    
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
        
        # Update storage settings with validation
        if 'max_snapshots' in data:
            try:
                max_snapshots = int(data['max_snapshots'])
                if max_snapshots < 1:
                    return jsonify({'status': 'error', 'message': 'max_snapshots must be greater than 0'}), 400
                config['max_snapshots'] = max_snapshots
            except (ValueError, TypeError):
                return jsonify({'status': 'error', 'message': 'Invalid max_snapshots value'}), 400
        
        if 'max_events' in data:
            try:
                max_events = int(data['max_events'])
                if max_events < 1:
                    return jsonify({'status': 'error', 'message': 'max_events must be greater than 0'}), 400
                config['max_events'] = max_events
            except (ValueError, TypeError):
                return jsonify({'status': 'error', 'message': 'Invalid max_events value'}), 400
        
        if 'auto_cleanup' in data:
            config['auto_cleanup'] = bool(data['auto_cleanup'])
        
        # Save settings to file
        if not settings_manager.update_settings(config):
            return jsonify({'status': 'error', 'message': 'Failed to save settings'}), 500
        
        return jsonify({'status': 'success', 'config': config})
        
    except Exception as e:
        print(f"Error updating storage settings: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/camera/status')
def api_camera_status():
    """API endpoint to get camera status"""
    global camera_running
    return jsonify({'running': camera_running})

@app.route('/api/camera/start', methods=['POST'])
def api_camera_start():
    """API endpoint to start the camera"""
    global camera_running, camera_thread, camera_cap
    
    if camera_running:
        return jsonify({'status': 'success', 'message': 'Camera is already running'})
    
    try:
        # Initialize camera
        camera_cap = cv2.VideoCapture(config['camera_id'])
        if not camera_cap.isOpened():
            return jsonify({'status': 'error', 'message': 'Failed to open camera'}), 500
        
        # Set camera properties
        camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['frame_width'])
        camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['frame_height'])
        camera_cap.set(cv2.CAP_PROP_FPS, config['fps'])
        
        # Start camera thread
        camera_running = True
        camera_thread = threading.Thread(target=camera_stream)
        camera_thread.daemon = True
        camera_thread.start()
        
        # Wait briefly to ensure camera is initialized
        time.sleep(0.5)
        
        return jsonify({'status': 'success', 'message': 'Camera started successfully'})
    except Exception as e:
        print(f"Error starting camera: {str(e)}")
        if camera_cap is not None:
            camera_cap.release()
            camera_cap = None
        camera_running = False
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/camera/stop', methods=['POST'])
def api_camera_stop():
    """API endpoint to stop the camera"""
    global camera_running, camera_thread, camera_cap, latest_frame
    
    if not camera_running:
        return jsonify({'status': 'success', 'message': 'Camera is already stopped'})
    
    try:
        # Stop camera thread
        camera_running = False
        if camera_thread and camera_thread.is_alive():
            camera_thread.join(timeout=1.0)
        
        # Release camera
        if camera_cap is not None:
            camera_cap.release()
            camera_cap = None
        
        # Clear latest frame
        latest_frame = None
        
        # Ensure camera is fully stopped
        stop_camera()
        
        return jsonify({'status': 'success', 'message': 'Camera stopped successfully'})
    except Exception as e:
        print(f"Error stopping camera: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/camera/feed')
def api_camera_feed():
    """Serve camera feed as JPEG image"""
    global latest_frame, last_error_log_time, error_log_interval
    
    current_time = time.time()
    
    # Add frame rate control
    if hasattr(api_camera_feed, 'last_frame_time'):
        time_since_last_frame = current_time - api_camera_feed.last_frame_time
        if time_since_last_frame < 0.033:  # Limit to ~30 FPS
            time.sleep(0.033 - time_since_last_frame)
    
    if not camera_running:
        if current_time - last_error_log_time.get('camera_not_running', 0) > error_log_interval:
            print("Camera feed requested but camera is not running")
            last_error_log_time['camera_not_running'] = current_time
        return jsonify({'error': 'Camera is not running'}), 404
    
    if latest_frame is None:
        if current_time - last_error_log_time.get('no_frame', 0) > error_log_interval:
            print("Camera feed requested but no frame available")
            last_error_log_time['no_frame'] = current_time
        return jsonify({'error': 'No frame available'}), 404
    
    try:
        # Validate frame dimensions
        if latest_frame.size == 0 or latest_frame.shape[0] == 0 or latest_frame.shape[1] == 0:
            if current_time - last_error_log_time.get('invalid_dimensions', 0) > error_log_interval:
                print("Invalid frame dimensions")
                last_error_log_time['invalid_dimensions'] = current_time
            return jsonify({'error': 'Invalid frame dimensions'}), 500
        
        # Optimize frame encoding
        try:
            # Use lower quality JPEG encoding for better performance
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            _, buffer = cv2.imencode('.jpg', latest_frame, encode_params)
            
            # Update last frame time
            api_camera_feed.last_frame_time = current_time
            
            return send_file(
                BytesIO(buffer.tobytes()),
                mimetype='image/jpeg'
            )
        except Exception as e:
            if current_time - last_error_log_time.get('encoding_error', 0) > error_log_interval:
                print(f"Error encoding frame: {str(e)}")
                last_error_log_time['encoding_error'] = current_time
            return jsonify({'error': 'Error encoding frame'}), 500
            
    except Exception as e:
        if current_time - last_error_log_time.get('general_error', 0) > error_log_interval:
            print(f"Error serving camera feed: {str(e)}")
            last_error_log_time['general_error'] = current_time
        return jsonify({'error': 'Error serving camera feed'}), 500

@app.route('/api/faces/upload', methods=['POST'])
def api_faces_upload():
    """API endpoint to upload a new face"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    
    file = request.files['file']
    name = request.form.get('name', '')
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    
    if name == '':
        return jsonify({'status': 'error', 'message': 'Name is required'}), 400
    
    # Save the file
    try:
        # Create directory if it doesn't exist
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        
        # Save the file
        filename = f"{name}.jpg"
        filepath = os.path.join(UPLOADS_DIR, filename)
        file.save(filepath)
        
        return jsonify({'status': 'success', 'message': 'Face uploaded successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/faces/delete', methods=['POST'])
def api_faces_delete():
    """API endpoint to delete a face"""
    data = request.json
    
    if 'filename' not in data:
        return jsonify({'status': 'error', 'message': 'Filename is required'}), 400
    
    filename = data['filename']
    filepath = os.path.join(UPLOADS_DIR, filename)
    
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'status': 'success', 'message': 'Face deleted successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/snapshots/delete', methods=['POST'])
def api_snapshots_delete():
    """API endpoint to delete a snapshot"""
    data = request.json
    
    if not data:
        return jsonify({'status': 'error', 'message': 'No data provided'}), 400
    
    # Get filename from either filepath or filename parameter
    if 'filepath' in data:
        # Extract filename from filepath (e.g., '/snapshot/motion/filename.jpg' -> 'filename.jpg')
        filepath = data['filepath']
        filename = os.path.basename(filepath)
    elif 'filename' in data:
        filename = data['filename']
    else:
        return jsonify({'status': 'error', 'message': 'Either filepath or filename is required'}), 400
    
    # Check both motion and unknown_faces directories
    motion_path = os.path.join(SNAPSHOTS_MOTION_DIR, filename)
    unknown_path = os.path.join(SNAPSHOTS_UNKNOWN_DIR, filename)
    
    try:
        if os.path.exists(motion_path):
            os.remove(motion_path)
            return jsonify({'status': 'success', 'message': 'Snapshot deleted successfully'})
        elif os.path.exists(unknown_path):
            os.remove(unknown_path)
            return jsonify({'status': 'success', 'message': 'Snapshot deleted successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
    except Exception as e:
        print(f"Error deleting snapshot: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/emails')
def emails():
    """Render the emails page"""
    email_logs = get_email_logs(50)  # Get last 50 email logs
    return render_template('emails.html', emails=email_logs)  # Pass as 'emails' instead of 'email_logs'

@app.route('/clear_events/<event_type>', methods=['POST'])
def clear_events_route(event_type):
    db = next(get_db())
    clear_events(db, event_type)
    return jsonify({'message': f'Cleared {event_type} events'})

@app.route('/api/events/clear', methods=['POST'])
def api_events_clear():
    """API endpoint to clear event logs"""
    try:
        data = request.json
        if not data or 'type' not in data:
            return jsonify({'status': 'error', 'message': 'Event type is required'}), 400
        
        event_type = data['type'].lower()
        
        # Map of event types to database tables
        event_types = {
            'known_faces': 'known_faces',
            'unknown_faces': 'unknown_faces',
            'motion': 'motion_events',
            'emails': 'email_logs',
            'email': 'email_logs',  # Add 'email' as an alias for 'emails'
            'all': 'all'  # Special case to clear all tables
        }
        
        if event_type not in event_types:
            return jsonify({'status': 'error', 'message': f'Invalid event type. Must be one of: {", ".join(event_types.keys())}'}), 400
        
        db = next(get_db())
        
        try:
            if event_type == 'all':
                # Clear all tables
                db.query(KnownFace).delete()
                db.query(UnknownFace).delete()
                db.query(MotionEvent).delete()
                db.query(EmailLog).delete()
                message = 'All event logs cleared successfully'
            else:
                # Clear specific table
                if event_type == 'known_faces':
                    db.query(KnownFace).delete()
                elif event_type == 'unknown_faces':
                    db.query(UnknownFace).delete()
                elif event_type == 'motion':
                    db.query(MotionEvent).delete()
                elif event_type in ['emails', 'email']:  # Handle both 'emails' and 'email'
                    db.query(EmailLog).delete()
                message = f'{event_type} events cleared successfully'
            
            db.commit()
            return jsonify({'status': 'success', 'message': message})
            
        except Exception as db_error:
            db.rollback()
            print(f"Database error while clearing events: {str(db_error)}")
            return jsonify({'status': 'error', 'message': 'Failed to clear events. Please try again.'}), 500
        
    except Exception as e:
        print(f"Error clearing events: {str(e)}")
        return jsonify({'status': 'error', 'message': 'An error occurred while clearing events. Please try again.'}), 500

@app.route('/api/snapshots/clear', methods=['POST'])
def api_snapshots_clear():
    """API endpoint to clear snapshots by type"""
    try:
        data = request.json
        if not data or 'type' not in data:
            return jsonify({'status': 'error', 'message': 'Snapshot type is required'}), 400
        
        snapshot_type = data['type'].lower()
        
        if snapshot_type == 'motion':
            directory = SNAPSHOTS_MOTION_DIR
        elif snapshot_type == 'unknown':
            directory = SNAPSHOTS_UNKNOWN_DIR
        else:
            return jsonify({'status': 'error', 'message': 'Invalid snapshot type. Must be either "motion" or "unknown"'}), 400
        
        # Delete all files in the directory
        count = 0
        for filename in os.listdir(directory):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(directory, filename)
                try:
                    os.remove(filepath)
                    count += 1
                except Exception as e:
                    print(f"Error deleting file {filepath}: {str(e)}")
        
        return jsonify({
            'status': 'success',
            'message': f'Cleared {count} {snapshot_type} snapshots'
        })
        
    except Exception as e:
        print(f"Error clearing snapshots: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/clear_all', methods=['POST'])
def api_clear_all():
    """API endpoint to clear all data (events and snapshots)"""
    try:
        db = next(get_db())
        
        # Clear all database tables
        try:
            db.query(KnownFace).delete()
            db.query(UnknownFace).delete()
            db.query(MotionEvent).delete()
            db.query(EmailLog).delete()
            db.commit()
        except Exception as db_error:
            db.rollback()
            print(f"Database error while clearing events: {str(db_error)}")
            return jsonify({'status': 'error', 'message': 'Failed to clear database records. Please try again.'}), 500
        
        # Clear all snapshot files
        try:
            # Clear motion snapshots
            if os.path.exists(SNAPSHOTS_MOTION_DIR):
                for filename in os.listdir(SNAPSHOTS_MOTION_DIR):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        filepath = os.path.join(SNAPSHOTS_MOTION_DIR, filename)
                        try:
                            os.remove(filepath)
                        except Exception as e:
                            print(f"Error deleting file {filepath}: {str(e)}")
            
            # Clear unknown face snapshots
            if os.path.exists(SNAPSHOTS_UNKNOWN_DIR):
                for filename in os.listdir(SNAPSHOTS_UNKNOWN_DIR):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        filepath = os.path.join(SNAPSHOTS_UNKNOWN_DIR, filename)
                        try:
                            os.remove(filepath)
                        except Exception as e:
                            print(f"Error deleting file {filepath}: {str(e)}")
        except Exception as e:
            print(f"Error clearing snapshots: {str(e)}")
            return jsonify({'status': 'error', 'message': 'Failed to clear snapshot files. Please try again.'}), 500
        
        return jsonify({
            'status': 'success',
            'message': 'All data cleared successfully'
        })
        
    except Exception as e:
        print(f"Error clearing all data: {str(e)}")
        return jsonify({'status': 'error', 'message': 'An error occurred while clearing data. Please try again.'}), 500

def process_frame(frame):
    global detector, config
    
    if detector is None:
        detector = detection.FaceDetector()
    
    # Process frame for face detection
    faces = detector.detect_faces(frame)
    
    # Get current date and time
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    
    db = next(get_db())
    
    # Check and perform automatic cleanup if enabled
    if config.get('auto_cleanup', False):
        try:
            # Clean up snapshots
            for directory in [SNAPSHOTS_MOTION_DIR, SNAPSHOTS_UNKNOWN_DIR]:
                if os.path.exists(directory):
                    files = sorted([f for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png'))],
                                 key=lambda x: os.path.getmtime(os.path.join(directory, x)))
                    while len(files) > config['max_snapshots']:
                        os.remove(os.path.join(directory, files.pop(0)))
            
            # Clean up events
            for model in [KnownFace, UnknownFace, MotionEvent, EmailLog]:
                count = db.query(model).count()
                if count > config['max_events']:
                    # Get the oldest events to delete
                    to_delete = db.query(model).order_by(model.id.asc()).limit(count - config['max_events']).all()
                    for event in to_delete:
                        db.delete(event)
                    db.commit()
        except Exception as e:
            print(f"Error during automatic cleanup: {str(e)}")
    
    for face in faces:
        name = face.get('name', 'Unknown')
        confidence = face.get('confidence', 0.0)
        face_img = face.get('image')
        
        if face_img is not None:
            # Save face image
            if name == 'Unknown':
                face_path = os.path.join(SNAPSHOTS_UNKNOWN_DIR, f"{date_str}_{time_str}.jpg")
            else:
                face_path = os.path.join(SNAPSHOTS_MOTION_DIR, f"{date_str}_{time_str}.jpg")
            
            cv2.imwrite(face_path, face_img)
            
            # Save to database
            if name == 'Unknown':
                unknown_face = UnknownFace(
                    date=date_str,
                    time=time_str,
                    confidence=confidence,
                    snapshot=face_path,
                    status='Detected'
                )
                db.add(unknown_face)
            else:
                known_face = KnownFace(
                    name=name,
                    date=date_str,
                    time=time_str,
                    confidence=confidence,
                    snapshot=face_path,
                    status='Detected'
                )
                db.add(known_face)
            
            db.commit()
    
    # Process frame for motion detection
    motion_detected = detector.detect_motion(frame)
    if motion_detected:
        motion_img = detector.get_motion_frame()
        if motion_img is not None:
            # Save motion image
            motion_path = os.path.join(SNAPSHOTS_MOTION_DIR, f"{date_str}_{time_str}.jpg")
            cv2.imwrite(motion_path, motion_img)
            
            # Save to database
            motion_event = MotionEvent(
                date=date_str,
                time=time_str,
                confidence=1.0,
                snapshot=motion_path,
                status='Detected'
            )
            db.add(motion_event)
            db.commit()
    
    return frame

def send_email_notification(subject, body, attachment_path=None):
    if not config['email_enabled']:
        return
    
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        from email.mime.image import MIMEImage
        
        msg = MIMEMultipart()
        msg['From'] = config['sender_email']
        msg['To'] = config['recipient_email']
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment_path))
                msg.attach(img)
        
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        server.starttls()
        server.login(config['smtp_username'], config['smtp_password'])
        server.send_message(msg)
        server.quit()
        
        # Log email to database
        db = next(get_db())
        email_log = EmailLog(
            date=datetime.now().strftime('%Y-%m-%d'),
            time=datetime.now().strftime('%H:%M:%S'),
            recipient=config['recipient_email'],
            subject=subject,
            status='Sent'
        )
        db.add(email_log)
        db.commit()
        
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        # Log failed email to database
        db = next(get_db())
        email_log = EmailLog(
            date=datetime.now().strftime('%Y-%m-%d'),
            time=datetime.now().strftime('%H:%M:%S'),
            recipient=config['recipient_email'],
            subject=subject,
            status=f'Failed: {str(e)}'
        )
        db.add(email_log)
        db.commit()

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(SNAPSHOTS_MOTION_DIR, exist_ok=True)
    os.makedirs(SNAPSHOTS_UNKNOWN_DIR, exist_ok=True)
    
    # Start the Flask app
    print("Starting Flask application...")
    print("Access the application at: http://127.0.0.1:5000")
    socketio.run(app, host='127.0.0.1', port=5000, debug=True) 