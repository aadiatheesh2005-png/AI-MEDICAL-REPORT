from flask import Flask, render_template, request, send_file
import joblib
import os

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/generate', methods=['POST'])
def generate():

    # Get form data
    age = request.form['age']
    gender = request.form['gender']
    height = request.form['height']
    weight = request.form['weight']
    blood_group = request.form['blood_group']
    bp = request.form['bp']
    temp = request.form['temp']

    # AI Prediction
    prediction = model.predict([[float(age), float(bp)]])
    diagnosis = prediction[0]

    # Save PDF inside temporary file
    pdf_path = "report.pdf"
    pdf = SimpleDocTemplate(pdf_path, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("<b>EVERGREEN WELLNESS HOSPITAL</b>", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("<b>MEDICAL REPORT</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.5 * inch))

    # Patient Information Table
    patient_data = [
        ["Age", age],
        ["Gender", gender],
        ["Height", height],
        ["Weight", weight],
        ["Blood Group", blood_group],
    ]

    patient_table = Table(patient_data, colWidths=[2*inch, 3*inch])
    patient_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTSIZE', (0,0), (-1,-1), 11),
    ]))

    elements.append(Paragraph("<b>Patient Information</b>", styles["Heading3"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(patient_table)
    elements.append(Spacer(1, 0.5 * inch))

    # Vital Signs Table
    vital_data = [
        ["Blood Pressure", bp],
        ["Body Temperature", temp],
    ]

    vital_table = Table(vital_data, colWidths=[2*inch, 3*inch])
    vital_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTSIZE', (0,0), (-1,-1), 11),
    ]))

    elements.append(Paragraph("<b>Vital Signs</b>", styles["Heading3"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(vital_table)
    elements.append(Spacer(1, 0.5 * inch))

    # Diagnosis
    elements.append(Paragraph("<b>Diagnosis</b>", styles["Heading3"]))
    elements.append(Spacer(1, 0.2 * inch))

    diagnosis_text = f"""
    Based on the clinical inputs and AI analysis,
    the predicted condition is: <b>{diagnosis}</b>.
    """

    elements.append(Paragraph(diagnosis_text, styles["Normal"]))
    elements.append(Spacer(1, 1 * inch))

    elements.append(Paragraph("Doctor Signature: ____________________", styles["Normal"]))

    # Build PDF
    pdf.build(elements)

    # Send file to user for download
    return send_file(pdf_path, as_attachment=True)


# IMPORTANT FOR RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)