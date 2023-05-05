import os
import pickle
import joblib
import numpy as np
from flask import Flask, render_template, request

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/result",methods=['POST','GET'])
def result():
    gender=int(request.form['gender'])
    age=int(request.form['age'])
    hypertension=int(request.form['hypertension'])
    heart_disease=int(request.form['heart_disease'])
    ever_married=int(request.form['ever_married'])
    work_type=int(request.form['work_type'])
    Residence_type=int(request.form['Residence_type'])
    avg_glucose_level=float(request.form['avg_glucose_level'])
    bmi=float(request.form['bmi'])
    smoking_status=int(request.form['smoking_status'])

    x=np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]).reshape(1,-1)

    scalar_path=os.path.join('/Users/LKorenfeld/Desktop/Stroke-prediction-main','models/scalar.pkl')
    scalar=None
    with open(scalar_path,'rb') as scalar_file:
        scalar=pickle.load(scalar_file)

    x=scalar.transform(x)

    model_path=os.path.join('/Users/LKorenfeld/Desktop/Stroke-prediction-main','models/lgbm.sav')
    lgbm=joblib.load(model_path)

    y_pred=lgbm.predict(x)

    if y_pred==0:
        return render_template('nostroke.html')
    else:
        return render_template('stroke.html')

@app.route("/informations")
def info():
    return render_template("informations.html")

if(__name__)=="__main__":
    app.run(debug=False)
