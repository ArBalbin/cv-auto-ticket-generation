#!/usr/bin/env python3
# app/app_final.py
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session, flash
import cv2
from ultralytics import YOLO
from inference.heatmap import make_heatmap
import threading
import time
import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import pooling
from flask_cors import CORS

from queue_tracker import QueueTracker, QueueZone
from auth_service import AuthService
from prediction_service import PredictionService

load_dotenv()

app = Flask(__name__, template_folder=str(project_root / 'app' / 'templates'))
app.secret_key = os.getenv('SECRET_KEY', '')

CORS(app,
     origins=['http://localhost:3000'],
     supports_credentials=True,
     allow_headers=['Content-Type', 'Authorization'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])

# ── DB pool ───────────────────────────────────────────────────────────────────
_db_cfg = {
    'host':     os.getenv('DB_HOST', 'localhost'),
    'port':     int(os.getenv('DB_PORT', '3306')),
    'database': os.getenv('DB_NAME', 'Crowd_Detection'),
    'user':     os.getenv('DB_USERNAME', 'crowd_monitoring_user'),
    'password': os.getenv('DB_PASSWORD', 'password123'),
}
try:
    db_pool = pooling.MySQLConnectionPool(
        pool_name="crowd_pool", pool_size=10, pool_reset_session=True, **_db_cfg)
    print("✅ DB pool created")
except mysql.connector.Error as err:
    print(f"❌ DB pool error: {err}")
    db_pool = None

# ── Auth service ──────────────────────────────────────────────────────────────
auth = AuthService(db_pool)

# ── Model ─────────────────────────────────────────────────────────────────────
try:
    model = YOLO('yolov8n.pt')
    print("✅ YOLO model loaded")
except Exception as e:
    try:
        model = YOLO(str(project_root / 'yolov8n.pt'))
        print("✅ YOLO model loaded (fallback path)")
    except Exception as e2:
        print(f"❌ YOLO load failed: {e2}")
        model = None

# ── Capture ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("⚠️  Camera not available — video feed will be unavailable")

# ── Queue setup ───────────────────────────────────────────────────────────────
queue_zone    = QueueZone(x1=100, y1=50, x2=540, y2=430)
queue_tracker = QueueTracker(zone=queue_zone)

QUEUE_CONFIG = {
    'avg_service_time': 3.0,
    'num_counters':     3,
}

current_data = {
    'count': 0, 'avg_density': 0.0, 'max_density': 0.0,
    'queue_length': 0, 'active_counters': QUEUE_CONFIG['num_counters'],
    'estimated_wait_time': 0.0, 'predicted_wait_5min': 0.0,
    'predicted_wait_15min': 0.0, 'predicted_wait_30min': 0.0,
    'system_utilization': 0.0, 'arrival_rate': 0.0,
    'service_rate': 1.0 / QUEUE_CONFIG['avg_service_time'],
    'timestamp': time.time(),
}
data_lock = threading.Lock()

# ── Prediction service ────────────────────────────────────────────────────────
predictor = PredictionService(QUEUE_CONFIG)


# ============================================================================
# VIDEO STREAM
# ============================================================================

def gen_frames():
    if model is None:
        print("❌ Cannot stream — YOLO model not loaded")
        return
    if not cap.isOpened():
        print("❌ Cannot stream — camera not available")
        return

    _fallback: dict = {}
    _next_id = [1000]
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def _fallback_id(cx, cy, b=80):
        key = (cx // b, cy // b)
        if key not in _fallback:
            _fallback[key] = _next_id[0]
            _next_id[0] += 1
        return _fallback[key]

    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️  Frame read failed — camera disconnected?")
            time.sleep(0.1)
            continue

        try:
            results = model.track(
                frame, imgsz=320, persist=True, conf=0.25, verbose=False)
        except Exception as e:
            print(f"⚠️  YOLO tracking error: {e}")
            continue

        # ── Parse YOLO detections ─────────────────────────────────────────────
        # We apply NMS-style deduplication here, before the queue tracker ever
        # sees the data, to handle cases where YOLO emits multiple overlapping
        # boxes for the same physical person (very common at close range).
        raw_detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                try:
                    if int(box.cls[0]) != 0:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    tid = int(box.id[0]) if box.id is not None else \
                          _fallback_id((x1 + x2) // 2, (y1 + y2) // 2)
                    conf = float(box.conf[0]) if box.id is not None else 0.5
                    raw_detections.append({'track_id': tid, 'bbox': (x1, y1, x2, y2), 'conf': conf})
                except (IndexError, ValueError) as e:
                    print(f"⚠️  Box parse error: {e}")
                    continue

        # Greedy NMS: keep highest-confidence box; suppress any later box
        # whose centre is within DEDUP_CENTRE_FRAC * avg_diagonal of a kept box.
        kept, persons, tracked_persons = [], [], []
        for det in sorted(raw_detections, key=lambda d: d['conf'], reverse=True):
            bbox = det['bbox']
            cx   = (bbox[0] + bbox[2]) / 2
            cy   = (bbox[1] + bbox[3]) / 2
            diag = ((bbox[2]-bbox[0])**2 + (bbox[3]-bbox[1])**2) ** 0.5

            duplicate = False
            for kb in kept:
                kcx = (kb[0] + kb[2]) / 2
                kcy = (kb[1] + kb[3]) / 2
                kd  = ((kb[2]-kb[0])**2 + (kb[3]-kb[1])**2) ** 0.5
                dist = ((cx-kcx)**2 + (cy-kcy)**2) ** 0.5
                # IoU check
                ix1, iy1 = max(bbox[0], kb[0]), max(bbox[1], kb[1])
                ix2, iy2 = min(bbox[2], kb[2]), min(bbox[3], kb[3])
                inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                union = (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) + (kb[2]-kb[0])*(kb[3]-kb[1]) - inter
                iou   = inter / union if union > 0 else 0.0
                if iou > 0.15 or dist < 0.55 * (diag + kd) / 2:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(bbox)
                persons.append(bbox)
                tracked_persons.append({'track_id': det['track_id'], 'bbox': bbox})
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        count = len(persons)
        h, w  = frame.shape[:2]
        avg_d, max_d = predictor.calculate_density(persons, w, h)

        predictor.update(count, current_data, data_lock)
        with data_lock:
            current_data.update({
                'count': count, 'avg_density': avg_d,
                'max_density': max_d, 'timestamp': time.time(),
            })

        try:
            queue_tracker.process_frame(tracked_persons, frame=frame)
        except Exception as e:
            print(f"⚠️  Queue tracker error: {e}")

        try:
            overlay = cv2.addWeighted(frame, 0.7, make_heatmap(frame, persons), 0.3, 0)
        except Exception as e:
            print(f"⚠️  Heatmap error: {e}")
            overlay = frame.copy()

        queue_tracker.draw_on_frame(overlay)

        text = f"Count: {count} | Density: {max_d:.1f}"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.65, 2)
        cv2.rectangle(overlay, (8, 8), (tw + 16, th + 16), (0, 0, 0), -1)
        cv2.putText(overlay, text, (12, th + 10), FONT, 0.65, (0, 255, 0), 2)

        ok, buf = cv2.imencode('.jpg', overlay)
        if not ok:
            continue
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'


# ============================================================================
# SESSION AUTH ROUTES
# ============================================================================

@app.route('/')
def home():
    return redirect(url_for('computer_vision_dashboard' if auth.is_authenticated() else 'login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if auth.is_authenticated():
        return redirect(url_for('computer_vision_dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        ok, err  = auth.login_user(username, password)
        if ok:
            flash('Login successful!', 'success')
            return redirect(url_for('computer_vision_dashboard'))
        flash(err, 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    auth.logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ============================================================================
# DASHBOARD ROUTES
# ============================================================================

@app.route('/dashboard/computer-vision')
@auth.require_session
def computer_vision_dashboard():
    return render_template('index.html', username=auth.current_user())

@app.route('/dashboard/queueflow')
@auth.require_session
def queueflow_dashboard():
    return render_template('queueflow_dashboard.html', username=auth.current_user())

@app.route('/video')
@auth.require_session
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================================================================
# DATA API (staff — requires login)
# ============================================================================

@app.route('/api/data')
@auth.require_api_auth
def get_data():
    with data_lock:
        return jsonify(current_data)

@app.route('/api/queue/adjust_counters', methods=['POST'])
@auth.require_api_auth
def adjust_counters():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid JSON body'}), 400
    nc = data.get('counters')
    if nc is None:
        return jsonify({'error': 'counters field is required'}), 400
    try:
        nc = int(nc)
    except (TypeError, ValueError):
        return jsonify({'error': 'counters must be an integer'}), 400
    if not (1 <= nc <= 10):
        return jsonify({'error': 'counters must be between 1 and 10'}), 400
    with data_lock:
        current_data['active_counters'] = nc
    return jsonify({'success': True, 'counters': nc})


# ============================================================================
# REACT AUTH API
# ============================================================================

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid JSON body'}), 400
    username = data.get('username', '').strip()
    password = data.get('password', '')
    ok, err  = auth.login_user(username, password)
    if ok:
        return jsonify({'access_token': 'jwt-token-placeholder',
                        'user': {'id': '1', 'username': username}}), 200
    return jsonify({'error': err}), 401

@app.route('/api/auth/me')
def api_get_user():
    if auth.is_authenticated():
        return jsonify({'userId': '1', 'username': auth.current_user()}), 200
    return jsonify({'error': 'Not authenticated'}), 401

@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    auth.logout_user()
    return jsonify({'message': 'Logged out successfully'}), 200


# ============================================================================
# REACT CROWD API
# ============================================================================

@app.route('/api/crowd/data')
def api_crowd_data():
    with data_lock:
        return jsonify({k: current_data[k]
                        for k in ('count', 'avg_density', 'max_density', 'timestamp')})

@app.route('/api/crowd/video')
def api_video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================================================================
# REACT QUEUE API
# ============================================================================

@app.route('/api/queue/data')
@auth.require_api_auth
def api_queue_data():
    with data_lock:
        data = dict(current_data)
    data.update(queue_tracker.get_state())
    return jsonify(data)

@app.route('/api/queue/list')
def api_queue_list():
    """Live queue list — PUBLIC (no login needed)."""
    return jsonify(queue_tracker.get_state())

@app.route('/api/queue/done', methods=['POST'])
@auth.require_api_auth
def api_queue_done():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid JSON body'}), 400
    qn = data.get('queue_number')
    if qn is None:
        return jsonify({'error': 'queue_number is required'}), 400
    try:
        qn = int(qn)
    except (TypeError, ValueError):
        return jsonify({'error': 'queue_number must be an integer'}), 400
    if queue_tracker.mark_transaction_done(qn):
        return jsonify({'success': True, 'message': f'Q{qn:03d} completed',
                        'queue_state': queue_tracker.get_state()})
    return jsonify({'success': False, 'error': f'Q{qn:03d} not found in active queue'}), 404

@app.route('/api/queue/zone', methods=['GET'])
def api_get_zone():
    z = queue_zone
    return jsonify({'x1': z.x1, 'y1': z.y1, 'x2': z.x2, 'y2': z.y2})

@app.route('/api/queue/zone', methods=['POST'])
@auth.require_api_auth
def api_set_zone():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid JSON body'}), 400
    try:
        queue_zone.set_zone(
            int(data['x1']), int(data['y1']),
            int(data['x2']), int(data['y2'])
        )
    except KeyError as e:
        return jsonify({'error': f'Missing field: {e}'}), 400
    except (TypeError, ValueError) as e:
        return jsonify({'error': f'Invalid value: {e}'}), 400
    z = queue_zone
    return jsonify({'success': True,
                    'zone': {'x1': z.x1, 'y1': z.y1, 'x2': z.x2, 'y2': z.y2}})

@app.route('/api/queue/reset', methods=['POST'])
@auth.require_api_auth
def api_queue_reset():
    global queue_tracker
    queue_tracker = QueueTracker(zone=queue_zone)
    return jsonify({'success': True, 'message': 'Queue reset successfully'})

@app.route('/api/queue/status')
def api_queue_status_public():
    """Public ticket-holder check. Use ?q=<number>&token=<token>"""
    qn    = request.args.get('q', type=int)
    token = request.args.get('token', type=str)
    if qn is None or not token:
        return jsonify({'error': 'Missing parameters. Use ?q=<number>&token=<token>'}), 400

    result = queue_tracker.lookup_by_token(qn, token)
    if result is None:
        return jsonify({'error': f'Q{qn:03d} not found or already served.'}), 404
    if result.get('error') == 'invalid_token':
        return jsonify({'error': 'Invalid token. Please check your ticket.'}), 403

    result.pop('access_token', None)
    return jsonify(result), 200

@app.route('/api/queue/noshow_alerts')
@auth.require_api_auth
def api_noshow_alerts():
    return jsonify({'alerts': queue_tracker.get_noshow_alerts()})

@app.route('/api/queue/appearance_log')
@auth.require_api_auth
def api_appearance_log():
    return jsonify({'rejections': queue_tracker.appearance_rejections})

@app.route('/api/queue/noshow_config', methods=['GET'])
@auth.require_api_auth
def api_get_noshow_config():
    return jsonify({'noshow_window_seconds': queue_tracker.NOSHOW_WINDOW_SECONDS})

@app.route('/api/queue/noshow_config', methods=['POST'])
@auth.require_api_auth
def api_set_noshow_config():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid JSON body'}), 400
    secs = data.get('seconds')
    if secs is None:
        return jsonify({'error': 'seconds field is required'}), 400
    try:
        secs = int(secs)
    except (TypeError, ValueError):
        return jsonify({'error': 'seconds must be an integer'}), 400
    if not (30 <= secs <= 300):
        return jsonify({'error': 'seconds must be between 30 and 300'}), 400
    queue_tracker.NOSHOW_WINDOW_SECONDS = secs
    return jsonify({'success': True, 'noshow_window_seconds': secs})


# ============================================================================
# GLOBAL ERROR HANDLERS
# ============================================================================

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request', 'message': str(e)}), 400

@app.errorhandler(401)
def unauthorized(e):
    return jsonify({'error': 'Unauthorized', 'message': str(e)}), 401

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found', 'message': str(e)}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed', 'message': str(e)}), 405

@app.errorhandler(500)
def internal_error(e):
    print(f"❌ Internal server error: {e}")
    return jsonify({'error': 'Internal server error',
                    'message': 'Something went wrong. Please try again.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)