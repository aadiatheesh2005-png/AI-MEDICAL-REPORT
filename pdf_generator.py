from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, Image as RLImage)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
import datetime, os

PRIMARY    = colors.HexColor('#1a2e4a')
ACCENT     = colors.HexColor('#0ea5e9')
LIGHT_BLUE = colors.HexColor('#e0f2fe')
SUCCESS    = colors.HexColor('#16a34a')
WARNING    = colors.HexColor('#d97706')
DANGER     = colors.HexColor('#dc2626')
GRAY       = colors.HexColor('#64748b')
LIGHT_GRAY = colors.HexColor('#f1f5f9')
ROW_ALT    = colors.HexColor('#f8fafc')
WHITE      = colors.white
BORDER     = colors.HexColor('#cbd5e1')


def _p(text, style):
    return Paragraph(str(text) if text is not None else '', style)


class MedicalReportPDF:
    def __init__(self, filepath):
        self.lm = self.rm = 1.8 * cm
        self.content_w = A4[0] - self.lm - self.rm   # ~493 pt
        self.doc = SimpleDocTemplate(
            filepath, pagesize=A4,
            leftMargin=self.lm, rightMargin=self.rm,
            topMargin=2*cm, bottomMargin=2*cm,
            title="MedAI Medical Report", author="MedAI System",
        )
        self.styles = getSampleStyleSheet()
        self._mk_styles()
        self.story = []

    def _mk_styles(self):
        def ps(name, **kw):
            return ParagraphStyle(name, parent=self.styles['Normal'], **kw)
        self.s = {
            'body':  ps('body',  fontSize=9.5, leading=14,
                        textColor=colors.HexColor('#1e293b')),
            'label': ps('label', fontSize=9, fontName='Helvetica-Bold', textColor=GRAY),
            'value': ps('value', fontSize=9.5, fontName='Helvetica-Bold', textColor=PRIMARY),
            'th':    ps('th',    fontSize=9, fontName='Helvetica-Bold', textColor=WHITE),
            'th_c':  ps('thc',   fontSize=9, fontName='Helvetica-Bold', textColor=WHITE,
                        alignment=TA_CENTER),
            'hw':    ps('hw',    fontSize=10, fontName='Helvetica-Bold', textColor=WHITE,
                        leading=16),
            'small': ps('sm',    fontSize=8, textColor=GRAY),
            'disc':  ps('disc',  fontSize=8, fontName='Helvetica-Oblique', textColor=GRAY,
                        alignment=TA_JUSTIFY, leading=12),
        }

    # ── tiny helpers ──────────────────────────────────────────────────────────
    def _tbl(self, data, widths, cmds):
        t = Table(data, colWidths=widths)
        t.setStyle(TableStyle(cmds))
        return t

    def _base_cmds(self, hdr=True):
        c = [
            ('VALIGN',       (0,0),(-1,-1),'MIDDLE'),
            ('TOPPADDING',   (0,0),(-1,-1), 6),
            ('BOTTOMPADDING',(0,0),(-1,-1), 6),
            ('LEFTPADDING',  (0,0),(-1,-1), 10),
            ('RIGHTPADDING', (0,0),(-1,-1), 10),
            ('BOX',          (0,0),(-1,-1), 0.5, BORDER),
            ('INNERGRID',    (0,0),(-1,-1), 0.3, BORDER),
        ]
        if hdr:
            c.append(('BACKGROUND',(0,0),(-1,0),ACCENT))
        return c

    def _stripe(self, n, offset=1):
        return [('BACKGROUND',(0,i+offset),(-1,i+offset),
                 WHITE if i%2==0 else ROW_ALT) for i in range(n)]

    def _bar(self, title):
        self.story.append(Spacer(1, 10))
        self.story.append(self._tbl(
            [[_p(f'  {title}', self.s['hw'])]],
            [self.content_w],
            [('BACKGROUND',(0,0),(-1,-1),PRIMARY),
             ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
             ('LEFTPADDING',(0,0),(-1,-1),10)]))
        self.story.append(Spacer(1, 8))

    # ── header ────────────────────────────────────────────────────────────────
    def _header(self, report_id, doctor, hospital, gen_by):
        W = self.content_w
        now = datetime.datetime.now()
        logo_s = ParagraphStyle('lg',parent=self.styles['Normal'],
                                fontSize=22,fontName='Helvetica-Bold',textColor=WHITE)
        sub_s  = ParagraphStyle('ls',parent=self.styles['Normal'],
                                fontSize=9,textColor=colors.HexColor('#bae6fd'))
        date_s = ParagraphStyle('ld',parent=self.styles['Normal'],
                                fontSize=8,textColor=WHITE,alignment=TA_RIGHT)
        self.story.append(self._tbl(
            [[_p('MedAI', logo_s),
              _p('Advanced Medical Intelligence System<br/>Powered by Random Forest AI', sub_s),
              _p(f'<b>Report ID:</b> {report_id}<br/>'
                 f'<b>Date:</b> {now.strftime("%B %d, %Y")}<br/>'
                 f'<b>Time:</b> {now.strftime("%H:%M")}', date_s)]],
            [W*0.18, W*0.52, W*0.30],
            [('BACKGROUND',(0,0),(-1,-1),PRIMARY),
             ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
             ('TOPPADDING',(0,0),(-1,-1),14),('BOTTOMPADDING',(0,0),(-1,-1),14),
             ('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12)]))
        self.story.append(Spacer(1,4))
        s2 = ParagraphStyle('sb',parent=self.styles['Normal'],fontSize=9,
                            textColor=colors.HexColor('#1e293b'))
        self.story.append(self._tbl(
            [[_p(f'<b>Hospital/Clinic:</b> {hospital}', s2),
              _p(f'<b>Attending Physician:</b> {doctor}', s2),
              _p(f'<b>Generated By:</b> {gen_by}', s2)]],
            [W/3, W/3, W/3],
            [('BACKGROUND',(0,0),(-1,-1),LIGHT_BLUE),
             ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
             ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
             ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),
             ('BOX',(0,0),(-1,-1),0.5,BORDER)]))
        self.story.append(Spacer(1,12))

    # ── patient info ──────────────────────────────────────────────────────────
    def _patient_info(self, d):
        self._bar('PATIENT INFORMATION')
        W = self.content_w
        lw, vw = W*0.18, W*0.32
        rows = [
            [_p('Full Name',      self.s['label']), _p(d.get('name','N/A'),  self.s['value']),
             _p('Age',            self.s['label']), _p(f"{d.get('age','N/A')} years", self.s['value'])],
            [_p('Gender',         self.s['label']), _p(str(d.get('gender','N/A')).capitalize(), self.s['value']),
             _p('Weight/Height',  self.s['label']), _p(f"{d.get('weight','N/A')} kg / {d.get('height','N/A')} cm", self.s['value'])],
            [_p('Physician',      self.s['label']), _p(d.get('doctor_name','N/A'), self.s['value']),
             _p('Hospital/Clinic',self.s['label']), _p(d.get('hospital','N/A'),   self.s['value'])],
        ]
        self.story.append(self._tbl(rows, [lw,vw,lw,vw],
            self._base_cmds(hdr=False)+self._stripe(3,0)))
        notes = [(k, d.get(k,'')) for k in ('symptoms','medical_history','medications') if d.get(k)]
        label_map = {'symptoms':'Symptoms / Complaints:','medical_history':'Past Medical History:','medications':'Current Medications:'}
        if notes:
            self.story.append(Spacer(1,8))
            nrows = [[_p(label_map[k],self.s['label']),_p(v,self.s['body'])] for k,v in notes]
            self.story.append(self._tbl(nrows,[W*0.25,W*0.75],
                [('VALIGN',(0,0),(-1,-1),'TOP'),
                 ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
                 ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),
                 ('BOX',(0,0),(-1,-1),0.5,BORDER),
                 ('INNERGRID',(0,0),(-1,-1),0.3,BORDER)]+self._stripe(len(nrows),0)))

    # ── vitals ────────────────────────────────────────────────────────────────
    def _vitals(self, d):
        self._bar('VITAL SIGNS & LABORATORY VALUES')
        W  = self.content_w
        hdr = [_p(t,self.s['th']) for t in ['Parameter','Value','Unit','Reference Range']]
        items = [
            ('Blood Pressure (Systolic)',  d.get('blood_pressure_sys','--'),'mmHg','90-120'),
            ('Blood Pressure (Diastolic)', d.get('blood_pressure_dia','--'),'mmHg','60-80'),
            ('Heart Rate',                 d.get('heart_rate','--'),        'bpm', '60-100'),
            ('Body Temperature',           d.get('temperature','--'),       'F',   '97-99'),
            ('Blood Glucose (Fasting)',     d.get('glucose','--'),           'mg/dL','70-99'),
            ('Total Cholesterol',           d.get('cholesterol','--'),       'mg/dL','< 200'),
            ('Hemoglobin',                  d.get('hemoglobin','--'),        'g/dL', '12-17'),
            ('Oxygen Saturation',           d.get('oxygen_saturation','--'),'%',   '95-100'),
        ]
        ref_s = ParagraphStyle('ref',parent=self.styles['Normal'],fontSize=9,textColor=SUCCESS)
        rows = [hdr]+[[_p(p,self.s['body']),_p(f'<b>{v}</b>',self.s['value']),
                       _p(u,self.s['body']),_p(r,ref_s)] for p,v,u,r in items]
        self.story.append(self._tbl(rows,[W*0.42,W*0.18,W*0.15,W*0.25],
            self._base_cmds()+self._stripe(len(items))))

    # ── AI diagnosis ──────────────────────────────────────────────────────────
    def _ai_diagnosis(self, pred):
        self._bar('AI DIAGNOSTIC ASSESSMENT')
        W  = self.content_w
        rl = pred['risk_level']
        risk_color = {'low':SUCCESS,'medium':WARNING,'high':DANGER}.get(rl, GRAY)
        risk_bg    = {'low':colors.HexColor('#dcfce7'),
                      'medium':colors.HexColor('#fef3c7'),
                      'high':colors.HexColor('#fee2e2')}.get(rl, LIGHT_GRAY)

        # Three-column info row (no nested tables — just paragraphs with line breaks)
        diag_lbl_s = ParagraphStyle('dll',parent=self.styles['Normal'],fontSize=9,
                                    fontName='Helvetica-Bold',textColor=GRAY)
        diag_val_s = ParagraphStyle('dlv',parent=self.styles['Normal'],fontSize=13,
                                    fontName='Helvetica-Bold',textColor=PRIMARY,leading=17)
        risk_lbl_s = ParagraphStyle('rll',parent=self.styles['Normal'],fontSize=9,
                                    fontName='Helvetica-Bold',textColor=GRAY,alignment=TA_CENTER)
        risk_val_s = ParagraphStyle('rlv',parent=self.styles['Normal'],fontSize=16,
                                    fontName='Helvetica-Bold',textColor=risk_color,alignment=TA_CENTER)
        conf_lbl_s = ParagraphStyle('cll',parent=self.styles['Normal'],fontSize=9,
                                    fontName='Helvetica-Bold',textColor=GRAY,alignment=TA_CENTER)
        conf_val_s = ParagraphStyle('clv',parent=self.styles['Normal'],fontSize=16,
                                    fontName='Helvetica-Bold',textColor=ACCENT,alignment=TA_CENTER)

        c1,c2,c3 = W*0.56, W*0.22, W*0.22
        diag_cell = [_p('Primary Diagnosis',        diag_lbl_s),
                     Spacer(1,4),
                     _p(pred['primary_diagnosis'],   diag_val_s)]
        risk_cell = [_p('Risk Level',                risk_lbl_s),
                     Spacer(1,4),
                     _p(rl.upper(),                  risk_val_s)]
        conf_cell = [_p('AI Confidence',             conf_lbl_s),
                     Spacer(1,4),
                     _p(f"{pred['confidence']}%",    conf_val_s)]

        # Put each cell's content in its own single-column table so paragraph
        # wrapping works correctly within the outer 3-col table.
        def cell_tbl(items, w, bg):
            t = Table([[item] for item in items], colWidths=[w-20])
            t.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,-1),bg),
                ('LEFTPADDING',(0,0),(-1,-1),0),
                ('RIGHTPADDING',(0,0),(-1,-1),0),
                ('TOPPADDING',(0,0),(-1,-1),2),
                ('BOTTOMPADDING',(0,0),(-1,-1),2),
            ]))
            return t

        outer = self._tbl(
            [[cell_tbl(diag_cell,c1,WHITE),
              cell_tbl(risk_cell,c2,risk_bg),
              cell_tbl(conf_cell,c3,LIGHT_BLUE)]],
            [c1,c2,c3],
            [('VALIGN',(0,0),(-1,-1),'MIDDLE'),
             ('TOPPADDING',(0,0),(-1,-1),12),('BOTTOMPADDING',(0,0),(-1,-1),12),
             ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),
             ('BOX',(0,0),(-1,-1),1,BORDER),
             ('INNERGRID',(0,0),(-1,-1),0.5,BORDER)])
        self.story.append(outer)
        self.story.append(Spacer(1,8))

        bmi_s = ParagraphStyle('bmi',parent=self.styles['Normal'],fontSize=9.5,
                               fontName='Helvetica-Bold',textColor=ACCENT)
        n = 6; cw = W/n
        self.story.append(self._tbl(
            [[_p('<b>BMI Score:</b>',     self.s['label']),
              _p(str(pred['bmi']),        bmi_s),
              _p('<b>BMI Category:</b>',  self.s['label']),
              _p(pred['bmi_category'],    self.s['value']),
              _p('<b>Model Accuracy:</b>',self.s['label']),
              _p(pred['model_accuracy'],  self.s['value'])]],
            [cw]*n,
            [('BACKGROUND',(0,0),(-1,-1),LIGHT_GRAY),
             ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
             ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
             ('LEFTPADDING',(0,0),(-1,-1),8),
             ('BOX',(0,0),(-1,-1),0.5,BORDER)]))

    # ── vitals analysis ───────────────────────────────────────────────────────
    def _vitals_analysis(self, va):
        self._bar('VITALS ANALYSIS REPORT')
        W = self.content_w
        lmap = {'blood_pressure':'Blood Pressure','heart_rate':'Heart Rate',
                'temperature':'Temperature','glucose':'Blood Glucose',
                'cholesterol':'Cholesterol','hemoglobin':'Hemoglobin',
                'oxygen_saturation':'O2 Saturation'}
        cmap = {'normal':SUCCESS,'caution':WARNING,'warning':WARNING,'critical':DANGER}
        imap = {
            ('blood_pressure','normal'):  'Within healthy range.',
            ('blood_pressure','caution'): 'Elevated – lifestyle modifications advised.',
            ('blood_pressure','warning'): 'Hypertension stage 1/2 – medical review needed.',
            ('blood_pressure','critical'):'Critical – immediate attention required.',
            ('heart_rate','normal'):      'Normal sinus rhythm.',
            ('heart_rate','warning'):     'Irregular rate – ECG recommended.',
            ('temperature','normal'):     'Afebrile, no signs of pyrexia.',
            ('temperature','caution'):    'Low-grade fever – monitor closely.',
            ('temperature','warning'):    'Fever detected – antipyretic therapy may be needed.',
            ('temperature','critical'):   'Abnormal temperature – urgent evaluation required.',
            ('glucose','normal'):         'Fasting glucose within normal limits.',
            ('glucose','caution'):        'Pre-diabetic range – dietary counselling recommended.',
            ('glucose','warning'):        'Diabetic range – HbA1c testing advised.',
            ('glucose','critical'):       'Hypoglycaemia – immediate glucose correction required.',
            ('cholesterol','normal'):     'Cholesterol within desirable range.',
            ('cholesterol','caution'):    'Borderline high – diet modification advised.',
            ('cholesterol','warning'):    'High cholesterol – lipid panel and statin evaluation.',
            ('hemoglobin','normal'):      'Within normal reference range.',
            ('hemoglobin','warning'):     'Anaemia detected – iron studies recommended.',
            ('hemoglobin','critical'):    'Severe anaemia – transfusion may be required.',
            ('oxygen_saturation','normal'):'Adequate peripheral O2 saturation.',
            ('oxygen_saturation','warning'):'Low SpO2 – supplemental O2 may be required.',
            ('oxygen_saturation','critical'):'Critical – immediate respiratory support.',
        }
        hdr  = [_p(t,self.s['th']) for t in ['Parameter','Status','Clinical Interpretation']]
        rows = [hdr]
        for key,(stxt,stype) in va.items():
            sc = cmap.get(stype, GRAY)
            ss = ParagraphStyle('ss'+key,parent=self.styles['Normal'],
                                fontSize=9.5,fontName='Helvetica-Bold',textColor=sc)
            rows.append([_p(lmap.get(key,key.replace('_',' ').title()),self.s['body']),
                         _p(stxt,ss),
                         _p(imap.get((key,stype),'Clinical review recommended.'),self.s['body'])])
        self.story.append(self._tbl(rows,[W*0.25,W*0.25,W*0.50],
            self._base_cmds()+self._stripe(len(va))))

    # ── risk factors ──────────────────────────────────────────────────────────
    def _risk_factors(self, rfs):
        self._bar('IDENTIFIED RISK FACTORS')
        W = self.content_w
        rows = []
        for i,f in enumerate(rfs):
            ok = 'No significant' in f
            ss = ParagraphStyle('sym'+str(i),parent=self.styles['Normal'],
                                fontSize=11,fontName='Helvetica-Bold',
                                textColor=SUCCESS if ok else WARNING)
            rows.append([_p('●' if ok else '▲',ss),_p(f,self.s['body'])])
        col0 = 22  # wide enough for 6+6 padding + bullet glyph
        self.story.append(self._tbl(rows,[col0, W-col0],
            [('VALIGN',(0,0),(-1,-1),'MIDDLE'),
             ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
             ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
             ('BOX',(0,0),(-1,-1),0.5,BORDER),
             ('INNERGRID',(0,0),(-1,-1),0.3,BORDER)]+self._stripe(len(rows),0)))
        self.story.append(HRFlowable(width=W,thickness=0.5,color=BORDER))

    # ── differential diagnosis ────────────────────────────────────────────────
    def _probability_table(self, cp):
        self._bar('DIFFERENTIAL DIAGNOSIS PROBABILITIES')
        W   = self.content_w
        hdr = [_p(t,self.s['th']) for t in ['Condition','Probability (%)','Likelihood']]
        rows= [hdr]
        for cond,prob in sorted(cp.items(),key=lambda x:x[1],reverse=True):
            lh = ('High' if prob>50 else 'Moderate' if prob>25
                  else 'Low' if prob>10 else 'Unlikely')
            rows.append([_p(cond,self.s['body']),
                         _p(f'<b>{prob}%</b>',self.s['value']),
                         _p(lh,self.s['body'])])
        self.story.append(self._tbl(rows,[W*0.50,W*0.25,W*0.25],
            self._base_cmds()+self._stripe(len(cp))))

    # ── feature importance ────────────────────────────────────────────────────
    def _feature_importance(self, tf):
        self._bar('AI MODEL - KEY PREDICTIVE FACTORS')
        W   = self.content_w
        hdr = [_p(t,self.s['th']) for t in ['Feature','Importance (%)','Influence']]
        rows= [hdr]
        for feat,imp in tf:
            lv = ('Very High' if imp>20 else 'High' if imp>12
                  else 'Moderate' if imp>8 else 'Low')
            rows.append([_p(feat,self.s['body']),
                         _p(f'{imp}%',self.s['value']),
                         _p(lv,self.s['body'])])
        self.story.append(self._tbl(rows,[W*0.50,W*0.25,W*0.25],
            self._base_cmds()+self._stripe(len(tf))))

    # ── recommendations ───────────────────────────────────────────────────────
    def _recommendations(self, recs):
        self._bar('CLINICAL RECOMMENDATIONS')
        W    = self.content_w
        ns   = ParagraphStyle('nss',parent=self.styles['Normal'],
                              fontSize=10,fontName='Helvetica-Bold',textColor=ACCENT)
        rows = [[_p(f'<b>{i+1}.</b>',ns),_p(r,self.s['body'])]
                for i,r in enumerate(recs)]
        col0 = 22
        self.story.append(self._tbl(rows,[col0,W-col0],
            [('VALIGN',(0,0),(-1,-1),'TOP'),
             ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
             ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
             ('BOX',(0,0),(-1,-1),0.5,BORDER),
             ('INNERGRID',(0,0),(-1,-1),0.3,BORDER)]+self._stripe(len(rows),0)))
        self.story.append(HRFlowable(width=W,thickness=0.5,color=BORDER))

    # ── medical images ─────────────────────────────────────────────────────────
    def _images(self, paths, note):
        if not paths: return
        self._bar('MEDICAL IMAGING ANALYSIS')
        self.story.append(_p(note, self.s['body']))
        self.story.append(Spacer(1,8))
        mw = self.content_w * 0.6
        mh = mw * 0.65
        for p in paths:
            if not os.path.exists(p): continue
            try:
                img   = RLImage(p, width=mw, height=mh)
                label = _p(f'<b>Image:</b> {os.path.basename(p)}', self.s['small'])
                self.story.append(self._tbl([[img],[label]],[mw],
                    [('ALIGN',(0,0),(-1,-1),'CENTER'),
                     ('BOX',(0,0),(-1,-1),0.5,BORDER),
                     ('TOPPADDING',(0,0),(-1,-1),8),
                     ('BOTTOMPADDING',(0,0),(-1,-1),8)]))
                self.story.append(Spacer(1,8))
            except Exception:
                pass

    # ── footer ────────────────────────────────────────────────────────────────
    def _footer(self, report_id):
        W = self.content_w
        self.story.append(Spacer(1,16))
        self.story.append(HRFlowable(width=W,thickness=1,color=PRIMARY))
        self.story.append(Spacer(1,6))
        disc = ("IMPORTANT MEDICAL DISCLAIMER: This report is generated by MedAI using "
                "Random Forest machine learning for preliminary screening purposes only. "
                "It does NOT constitute a definitive diagnosis or substitute for professional "
                "medical advice. All findings must be validated by a qualified healthcare "
                "professional. MedAI assumes no liability for clinical decisions made solely "
                "on this AI-generated report.")
        self.story.append(_p(disc, self.s['disc']))
        self.story.append(Spacer(1,6))
        now_s = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")
        fc = ParagraphStyle('fc',parent=self.styles['Normal'],fontSize=8,
                            textColor=GRAY,alignment=TA_CENTER)
        fr = ParagraphStyle('fr',parent=self.styles['Normal'],fontSize=8,
                            textColor=GRAY,alignment=TA_RIGHT)
        self.story.append(self._tbl(
            [[_p(f'Report ID: {report_id}',self.s['small']),
              _p(f'Generated: {now_s}',fc),
              _p('MedAI System v2.0  |  Confidential',fr)]],
            [W/3,W/3,W/3],
            [('VALIGN',(0,0),(-1,-1),'MIDDLE')]))

    # ── entry ─────────────────────────────────────────────────────────────────
    def build(self, patient_data, prediction, image_paths, report_id, generated_by):
        pd = patient_data
        self._header(report_id, pd.get('doctor_name','N/A'),
                     pd.get('hospital','MedAI Hospital'), generated_by)
        self._patient_info(pd)
        self._vitals(pd)
        self._ai_diagnosis(prediction)
        self._vitals_analysis(prediction['vitals_analysis'])
        self._risk_factors(prediction['risk_factors'])
        self._probability_table(prediction['class_probabilities'])
        self._feature_importance(prediction['top_features'])
        self._recommendations(prediction['recommendations'])
        if image_paths and prediction.get('image_analysis'):
            self._images(image_paths, prediction['image_analysis'])
        self._footer(report_id)
        self.doc.build(self.story)


def generate_medical_pdf(patient_data, prediction, image_paths,
                         output_path, report_id, generated_by='System'):
    MedicalReportPDF(output_path).build(
        patient_data, prediction, image_paths, report_id, generated_by or 'System')
