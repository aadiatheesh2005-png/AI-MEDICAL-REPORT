from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os, json, uuid, datetime, traceback

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'medai-secret-key-2024')

# ── Vercel-compatible paths: use /tmp for all writes ──────────────────────────
TMP            = '/tmp'
UPLOAD_FOLDER  = os.path.join(TMP, 'medai_uploads')
REPORTS_FOLDER = os.path.join(TMP, 'medai_reports')
USERS_FILE     = os.path.join(TMP, 'users.json')
REPORTS_META_FILE = os.path.join(TMP, 'reports_meta.json')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# ── Lazy-load heavy deps so cold-start errors surface as JSON ─────────────────
_model = None
def get_model():
    global _model
    if _model is None:
        from ml_model import MedicalAIModel
        _model = MedicalAIModel()
    return _model

# ── JSON file helpers ─────────────────────────────────────────────────────────
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    demo_users = {
        'demo@medai.com': {
            'id': 'demo-user-id',
            'name': 'Demo Doctor',
            'email': 'demo@medai.com',
            'password': generate_password_hash('demo1234'),
            'created_at': datetime.datetime.now().isoformat()
        }
    }
    save_users(demo_users)
    return demo_users

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def load_reports_meta():
    if os.path.exists(REPORTS_META_FILE):
        with open(REPORTS_META_FILE) as f:
            return json.load(f)
    return {}

def save_reports_meta(meta):
    with open(REPORTS_META_FILE, 'w') as f:
        json.dump(meta, f)

# ── Auth helpers ──────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            if request.path.startswith('/api/'):
                return jsonify({'success': False, 'message': 'Authentication required'}), 401
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated

# ── Page routes ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login_page'))

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/signup')
def signup_page():
    return render_template('signup.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=session.get('username'))

@app.route('/generate')
@login_required
def generate_page():
    return render_template('generate.html', username=session.get('username'))

@app.route('/reports')
@login_required
def reports_page():
    meta = load_reports_meta()
    user_reports = [v for v in meta.values() if v.get('user_id') == session['user_id']]
    user_reports.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return render_template('reports.html', username=session.get('username'), reports=user_reports)

# ── API routes ────────────────────────────────────────────────────────────────
@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': 'Invalid request'}), 400
        users = load_users()
        if data.get('email') in users:
            return jsonify({'success': False, 'message': 'Email already registered'}), 409
        uid = str(uuid.uuid4())
        users[data['email']] = {
            'id': uid,
            'name': data['name'],
            'email': data['email'],
            'password': generate_password_hash(data['password']),
            'created_at': datetime.datetime.now().isoformat()
        }
        save_users(users)
        return jsonify({'success': True, 'message': 'Account created successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': 'Invalid request'}), 400
        users = load_users()
        user = users.get(data.get('email', ''))
        if not user or not check_password_hash(user['password'], data.get('password', '')):
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
        session['user_id'] = user['id']
        session['username'] = user['name']
        session['email'] = data['email']
        return jsonify({'success': True, 'message': 'Login successful'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/api/analyze', methods=['POST'])
@login_required
def analyze():
    try:
        patient_data = {
            'name':               request.form.get('patient_name', 'Unknown'),
            'age':                request.form.get('age', ''),
            'gender':             request.form.get('gender', ''),
            'weight':             request.form.get('weight', ''),
            'height':             request.form.get('height', ''),
            'blood_pressure_sys': request.form.get('blood_pressure_sys', ''),
            'blood_pressure_dia': request.form.get('blood_pressure_dia', ''),
            'heart_rate':         request.form.get('heart_rate', ''),
            'temperature':        request.form.get('temperature', ''),
            'glucose':            request.form.get('glucose', ''),
            'cholesterol':        request.form.get('cholesterol', ''),
            'hemoglobin':         request.form.get('hemoglobin', ''),
            'oxygen_saturation':  request.form.get('oxygen_saturation', ''),
            'symptoms':           request.form.get('symptoms', ''),
            'medical_history':    request.form.get('medical_history', ''),
            'medications':        request.form.get('medications', ''),
            'doctor_name':        request.form.get('doctor_name', 'Dr. AI System'),
            'hospital':           request.form.get('hospital', 'MedAI Hospital'),
        }

        image_paths = []
        if 'images' in request.files:
            files = request.files.getlist('images')
            for f in files:
                if f and allowed_file(f.filename):
                    fname = secure_filename(f.filename)
                    unique_name = f'{uuid.uuid4()}_{fname}'
                    fpath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
                    f.save(fpath)
                    image_paths.append(fpath)

        model = get_model()
        prediction = model.predict(patient_data, image_paths)

        from pdf_generator import generate_medical_pdf
        report_id    = str(uuid.uuid4())[:8].upper()
        pdf_filename = f'report_{report_id}.pdf'
        pdf_path     = os.path.join(REPORTS_FOLDER, pdf_filename)
        generate_medical_pdf(patient_data, prediction, image_paths,
                             pdf_path, report_id, session.get('username'))

        meta = load_reports_meta()
        meta[report_id] = {
            'report_id':    report_id,
            'user_id':      session['user_id'],
            'patient_name': patient_data['name'],
            'created_at':   datetime.datetime.now().isoformat(),
            'pdf_file':     pdf_filename,
            'risk_level':   prediction['risk_level'],
            'diagnosis':    prediction['primary_diagnosis'],
        }
        save_reports_meta(meta)

        return jsonify({
            'success':    True,
            'report_id':  report_id,
            'prediction': prediction,
            'pdf_url':    f'/api/download/{report_id}',
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/reports_data')
@login_required
def reports_data():
    try:
        meta = load_reports_meta()
        user_reports = [v for v in meta.values() if v.get('user_id') == session['user_id']]
        user_reports.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        total  = len(user_reports)
        low    = sum(1 for r in user_reports if r.get('risk_level') == 'low')
        medium = sum(1 for r in user_reports if r.get('risk_level') == 'medium')
        high   = sum(1 for r in user_reports if r.get('risk_level') == 'high')
        return jsonify({
            'total': total, 'low': low, 'medium': medium, 'high': high,
            'reports': user_reports[:10],
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download/<report_id>')
@login_required
def download_report(report_id):
    try:
        meta   = load_reports_meta()
        report = meta.get(report_id)
        if not report or report['user_id'] != session['user_id']:
            return jsonify({'error': 'Not found'}), 404
        pdf_path = os.path.join(REPORTS_FOLDER, report['pdf_file'])
        if not os.path.exists(pdf_path):
            return jsonify({'error': 'PDF expired — please regenerate the report'}), 404
        return send_file(pdf_path, as_attachment=True,
                         download_name=f'MedAI_Report_{report_id}.pdf')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Global error handlers ─────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return render_template('login.html'), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error', 'detail': str(e)}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    traceback.print_exc()
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': str(e)}), 500
    return render_template('login.html'), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)