from flask import Flask
from flask import request, render_template

from joblib import load
import numpy as np
import pandas as pd

app = Flask(__name__)

#TODO: load iris classfier from file
#  use try-except, print message and exit if there is a problem
try: 
    model = load("classify_model.sav")
except: 
    print("There was a problem laoding the iris classifier model from the file")



@app.route('/')
def index():
    return render_template('prediction_input.html')

#GET REQUEST
@app.route('/iris_prediction')
def get_iris_prediction():
    
    values_ok = True
    pred_str = 'None'
    pred_proba = 0.0
    
    #TODO: Get feature values as float from request.values dictionary
    #      Set values_ok to False if any conversion produces an error.
    featval = request.values

    try:
        sepal_length= float(featval['sepal_length'])
        sepal_width= float(featval['sepal_width'])
        petal_length= float(featval['petal_length'])
        petal_width= float(featval['petal_width'])
    except:
        values_ok = False

    if values_ok == True:

        values = np.array([[sepal_length,sepal_width,petal_length, petal_width]])

            #TODO: call predict() on the loaded classifier using the feature values
        #      and retrieve the predicted iris flower string
        #      assign string to pred_str
        ps = model.predict(values)[0]
        print(ps)
        if ps == 0:
            pred_str = 'setosa'
        elif ps == 1:
            pred_str = 'versicolor'
        elif ps == 2:
            pred_str = 'virginica'
        else: pred_str = 'None'


                #TODO: call predict_proba() on the loaded classifier
        #      assign probablity to pred_proba

        pred_proba = model.predict_proba(values)[0][ps]


    else: 
        values = ' There was a problem with measurements submitted. Check values above'



    
    
    
    return render_template('prediction_response.html',
                           request_dict=request.values,
                          pred_str=pred_str,
                           pred_proba = '{:.3f}'.format(pred_proba),
                          values_ok=values_ok) 


if __name__ == '__main__':
    app.debug = True
    app.run()
