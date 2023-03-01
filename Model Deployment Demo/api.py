# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import json
import sys
# Your API definition
app = Flask(__name__)

"""
http   database-operation
GET--->READ
PUT---->UPDATE
POST---->WRITE
DELETE ---> DELETE
"""








@app.route('/', methods=['GET'])
def predict():

    # if model file is loaded
    if model:
        try:

            print("hello")
            
            json_ = json.load(open("data.json")) #input data (from the front-end UI in json format)
            
            print(json_)


            #encoding! : GENDER : MALE, FEMALE-----> 0,   1
            query = pd.get_dummies(pd.DataFrame(json_)) #encoding

            #write the column names 
            query = query.reindex(columns=model_columns, fill_value=0)

            #make a prediction and save it a list
            prediction = list(model.predict(query)) #make predictions on input

            ### RETURN THE RESPONSE IN JSON FORMAT!!!!!!!!!!
            if prediction:
                ans = []
                for entry in prediction:
                    if entry:
                        ans.append( {'prediction': "Loan should be Accepted for this user!"}) #return predictions
                    else:
                        ans.append( {'prediction': "Loan should be Rejected for this user!"})

                return jsonify(ans)    
            

        except:

            return ( jsonify({'trace': traceback.format_exc()}) )
    else:
        print ('Train the model first')
        return ('No model here to use')











if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    model = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)