from flask import Flask, render_template, request, send_file
import joblib
import numpy as np
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

# load trained model
model = joblib.load("model.pkl")

# create reports folder if not exists
if not os.path.exists("reports"):
    os.makedirs("reports")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Patient Details
    age = int(request.form["age"])
    gender = int(request.form["gender"])
    height = float(request.form["height"])
    weight = float(request.form["weight"])

    # Symptoms
    fever = int(request.form["fever"])
    cough = int(request.form["cough"])
    chest_pain = int(request.form["chest_pain"])
    headache = int(request.form["headache"])
    nausea = int(request.form["nausea"])
    fatigue = int(request.form["fatigue"])

    # Vital Signs
    bp = float(request.form["bp"])
    temp = float(request.form["temp"])
    oxygen = float(request.form["oxygen"])
    heart_rate = float(request.form["heart_rate"])

    # Lab Results
    sugar = float(request.form["sugar"])
    hb = float(request.form["hb"])
    wbc = float(request.form["wbc"])
    platelets = float(request.form["platelets"])
    chol = float(request.form["chol"])
    creat = float(request.form["creat"])

    # ML model input
    features = np.array([[age, gender, height, weight, fever, cough, chest_pain,
                          headache, nausea, fatigue, bp, temp, oxygen,
                          heart_rate, sugar, hb, wbc, platelets, chol, creat]])

    prediction = model.predict(features)

    # convert prediction to text
    diagnosis = "Heart Disease Risk" if prediction[0] == 1 else "No Heart Disease Risk"

    generate_report(age, gender, height, weight,
                    fever, cough, chest_pain, headache, nausea, fatigue,
                    bp, temp, oxygen, heart_rate,
                    sugar, hb, wbc, platelets, chol, creat,
                    diagnosis)

    return send_file("reports/report.pdf", as_attachment=True)


def generate_report(age, gender, height, weight,
                    fever, cough, chest_pain, headache, nausea, fatigue,
                    bp, temp, oxygen, heart_rate,
                    sugar, hb, wbc, platelets, chol, creat,
                    diagnosis):

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI MEDICAL REPORT", styles['Title']))
    story.append(Spacer(1, 20))

    # Patient Details
    story.append(Paragraph("<b>Patient Details</b>", styles['Heading2']))
    story.append(Paragraph(f"Age: {age}", styles['Normal']))
    story.append(Paragraph(f"Gender: {gender}", styles['Normal']))
    story.append(Paragraph(f"Height: {height}", styles['Normal']))
    story.append(Paragraph(f"Weight: {weight}", styles['Normal']))
    story.append(Spacer(1, 15))

    # Symptoms
    story.append(Paragraph("<b>Symptoms</b>", styles['Heading2']))
    story.append(Paragraph(f"Fever: {fever}", styles['Normal']))
    story.append(Paragraph(f"Cough: {cough}", styles['Normal']))
    story.append(Paragraph(f"Chest Pain: {chest_pain}", styles['Normal']))
    story.append(Paragraph(f"Headache: {headache}", styles['Normal']))
    story.append(Paragraph(f"Nausea: {nausea}", styles['Normal']))
    story.append(Paragraph(f"Fatigue: {fatigue}", styles['Normal']))
    story.append(Spacer(1, 15))

    # Vital Signs
    story.append(Paragraph("<b>Vital Signs</b>", styles['Heading2']))
    story.append(Paragraph(f"Blood Pressure: {bp}", styles['Normal']))
    story.append(Paragraph(f"Temperature: {temp}", styles['Normal']))
    story.append(Paragraph(f"Oxygen Level: {oxygen}", styles['Normal']))
    story.append(Paragraph(f"Heart Rate: {heart_rate}", styles['Normal']))
    story.append(Spacer(1, 15))

    # Lab Results
    story.append(Paragraph("<b>Laboratory Results</b>", styles['Heading2']))
    story.append(Paragraph(f"Blood Sugar: {sugar}", styles['Normal']))
    story.append(Paragraph(f"Hemoglobin: {hb}", styles['Normal']))
    story.append(Paragraph(f"WBC Count: {wbc}", styles['Normal']))
    story.append(Paragraph(f"Platelets: {platelets}", styles['Normal']))
    story.append(Paragraph(f"Cholesterol: {chol}", styles['Normal']))
    story.append(Paragraph(f"Creatinine: {creat}", styles['Normal']))
    story.append(Spacer(1, 15))

    # AI Diagnosis
    story.append(Paragraph("<b>AI Diagnosis</b>", styles['Heading2']))
    story.append(Paragraph(f"Result: {diagnosis}", styles['Normal']))
    story.append(Spacer(1, 15))

    # Recommendation
    story.append(Paragraph("<b>Recommendation</b>", styles['Heading2']))
    story.append(Paragraph("Please consult a certified doctor for further medical evaluation.", styles['Normal']))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Doctor Signature: ____________________", styles['Normal']))

    doc = SimpleDocTemplate("reports/report.pdf")
    doc.build(story)


if __name__ == "__main__":
    app.run(debug=True)
