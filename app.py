from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os, json, uuid, datetime
from ml_model import MedicalAIModel
from pdf_generator import generate_medical_pdf

app = Flask(__name__)
app.secret_key = 'medai-secret-key-2024'
UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

USERS_FILE = 'users.json'
REPORTS_META_FILE = 'reports_meta.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

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

model = MedicalAIModel()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated

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

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    users = load_users()
    if data['email'] in users:
        return jsonify({'success': False, 'message': 'Email already registered'}), 400
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

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    users = load_users()
    user = users.get(data['email'])
    if not user or not check_password_hash(user['password'], data['password']):
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    session['user_id'] = user['id']
    session['username'] = user['name']
    session['email'] = data['email']
    return jsonify({'success': True, 'message': 'Login successful'})

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/api/analyze', methods=['POST'])
@login_required
def analyze():
    patient_data = {
        'name': request.form.get('patient_name', 'Unknown'),
        'age': request.form.get('age', ''),
        'gender': request.form.get('gender', ''),
        'weight': request.form.get('weight', ''),
        'height': request.form.get('height', ''),
        'blood_pressure_sys': request.form.get('blood_pressure_sys', ''),
        'blood_pressure_dia': request.form.get('blood_pressure_dia', ''),
        'heart_rate': request.form.get('heart_rate', ''),
        'temperature': request.form.get('temperature', ''),
        'glucose': request.form.get('glucose', ''),
        'cholesterol': request.form.get('cholesterol', ''),
        'hemoglobin': request.form.get('hemoglobin', ''),
        'oxygen_saturation': request.form.get('oxygen_saturation', ''),
        'symptoms': request.form.get('symptoms', ''),
        'medical_history': request.form.get('medical_history', ''),
        'medications': request.form.get('medications', ''),
        'doctor_name': request.form.get('doctor_name', 'Dr. AI System'),
        'hospital': request.form.get('hospital', 'MedAI Hospital'),
    }

    image_paths = []
    if 'images' in request.files:
        files = request.files.getlist('images')
        for f in files:
            if f and allowed_file(f.filename):
                fname = secure_filename(f.filename)
                unique_name = f"{uuid.uuid4()}_{fname}"
                fpath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
                f.save(fpath)
                image_paths.append(fpath)

    prediction = model.predict(patient_data, image_paths)

    report_id = str(uuid.uuid4())[:8].upper()
    pdf_filename = f"report_{report_id}.pdf"
    pdf_path = os.path.join(REPORTS_FOLDER, pdf_filename)

    generate_medical_pdf(patient_data, prediction, image_paths, pdf_path, report_id, session.get('username'))

    meta = load_reports_meta()
    meta[report_id] = {
        'report_id': report_id,
        'user_id': session['user_id'],
        'patient_name': patient_data['name'],
        'created_at': datetime.datetime.now().isoformat(),
        'pdf_file': pdf_filename,
        'risk_level': prediction['risk_level'],
        'diagnosis': prediction['primary_diagnosis']
    }
    save_reports_meta(meta)

    return jsonify({
        'success': True,
        'report_id': report_id,
        'prediction': prediction,
        'pdf_url': f'/api/download/{report_id}'
    })

@app.route('/api/reports_data')
@login_required
def reports_data():
    meta = load_reports_meta()
    user_reports = [v for v in meta.values() if v.get('user_id') == session['user_id']]
    user_reports.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    total = len(user_reports)
    low = sum(1 for r in user_reports if r.get('risk_level') == 'low')
    medium = sum(1 for r in user_reports if r.get('risk_level') == 'medium')
    high = sum(1 for r in user_reports if r.get('risk_level') == 'high')
    return jsonify({
        'total': total, 'low': low, 'medium': medium, 'high': high,
        'reports': user_reports[:10]
    })

@app.route('/api/download/<report_id>')
@login_required
def download_report(report_id):
    meta = load_reports_meta()
    report = meta.get(report_id)
    if not report or report['user_id'] != session['user_id']:
        return jsonify({'error': 'Not found'}), 404
    pdf_path = os.path.join(REPORTS_FOLDER, report['pdf_file'])
    return send_file(pdf_path, as_attachment=True, download_name=f"MedAI_Report_{report_id}.pdf")

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(REPORTS_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)
