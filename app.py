# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 12:25:30 2020

@author: LSREENI
"""

import numpy as np
import pandas as pd
from flask import Flask,request, render_template
import jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('LoanApprovalPrediction_PipeFinal.pkl', 'rb'))
#LoanModel_ppl = pickle.load(open('LoanApprovalPrediction_PipeFinal.pkl','rb'))
feature_names=['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education','Self_Employed','ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']

#print(LoanModel_ppl.predict(new_vals))
@app.route('/')
def home():
    return render_template('index.html')
    #return "<h1> Welcome to the Loan Application  </h1>"

@app.route('/predict',methods=['POST'])

def predict():
   # labels = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education','Self_Employed','ApplicantIncome', 
    #            'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    
    Loan_ID = request.form['Loan_ID']      
    Gender = request.form['Gender']   
    Married = request.form['Married']   
    Dependents = request.form['Dependents']  
    Education = request.form['Education']      
    Self_Employed = request.form['Self_Employed']   
    Property_Area = request.form['Property_Area']   
    ApplicantIncome = request.form['ApplicantIncome']  
    CoapplicantIncome = request.form['CoapplicantIncome']      
    LoanAmount = request.form['LoanAmount']   
    Loan_Amount_Term = request.form['Loan_Amount_Term']   
    Credit_History = request.form['Credit_History']  
    
    #features=request.get_json()
    #print(request.form.values())
    features = [x for x in request.form.values()]
    final_features=np.array(features)
    data_unseen = pd.DataFrame([final_features],columns = feature_names)
    prediction = model.predict(data_unseen)
    #final_features =  pd.DataFrame([[(features)]])
    # x = pd.DataFrame(data=data['data'], columns=data['feature_names'])
    values=[Loan_ID,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]
    print(values)
    
    x=pd.DataFrame(data=[values],columns=feature_names)
    new_vals= pd.DataFrame([['LP001116','Male','No','0','Not Graduate','No',3748,1668.0,110.0,360.0,1.0,'Semiurban']],columns = feature_names)
    #prediction = model.predict(x)
    output = prediction[0]
    #return jsonify(output)
    return render_template('index.html',prediction_text='LoanApporval = {}'.format(output))


if __name__=='__main__':
    app.run(debug=True)
