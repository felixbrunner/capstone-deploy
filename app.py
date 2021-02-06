### IMPORTS ###

import os
import joblib
import json
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField, BooleanField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

# import sys
# sys.path.append('../')

import src
from src.modeling import (
    coordinate_dict,
    fill_coordinates_with_station_means,
    grid_to_category,
    extract_datetime_features)



### UNPICKLE THE PREVIOUSLY-TRAINED SKLEARN MODEL ###

with open('columns.json') as fh:
    columns = json.load(fh)

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

pipeline = joblib.load('pipeline.pickle')



### SET UP DATABASE ###
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')
#DB = SqliteDatabase('predictions.db')

class Prediction(Model):
    observation_id = TextField(unique=True)
    request = TextField()
    predicted_outcome = IntegerField(null=True)
    true_outcome = IntegerField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)



### CREATE FLASK WEBSERVER ENDPOINTS ###

app = Flask(__name__)


# SHOULD_SEARCH ENDPOINT

@app.route('/should_search/', methods=['POST'])
def predict():
    """
    Produce authorisation permits for request
    
    Inputs:
        request: dictionary with format described below
        
        ```
        {
             "observation_id": <string>,
             "Type": <string>,
             "Date": <string>,
             "Part of a policing operation": <boolean>,
             "Latitude": <float>,
             "Longitude": <float>,
             "Gender": <string>,
             "Age range": <string>,
             "Officer-defined ethnicity": <string>,
             "Legislation": <string>,
             "Object of search": <string>,
             "station": <string>
        }
        ```
     
    Returns: 
        A dictionary with the outcome of the authorisation request.
                
        ```
        {
            "outcome": <boolean>
        }
                
                
        ```
        or 
        ```
        {
            "error": "some error message"
        }
        ```
        
    If the observation ID already exists on the database, it should return an error message.
    """

    req = request.get_json()
    
    ### DATA CHECKS ###
    
    # check if request has observation_id
    if 'observation_id' not in req:
        response = {'error': 'Must supply observation_id'}
        return jsonify(response)
    
    # check if request data has all necessary columns
    necessary_columns = {'observation_id',
                         'Type',
                         'Date',
                         'Part of a policing operation',
                         'Latitude',
                         'Longitude',
                         'Gender',
                         'Age range',
                         'Officer-defined ethnicity',
                         'Legislation',
                         'Object of search',
                         'station'}
    actual_columns = set(req.keys())
    
    if not necessary_columns.issubset(actual_columns):
        missing_columns = necessary_columns - actual_columns
        response = {'error': 'Missing columns: {}'.format(missing_columns)}
        return jsonify(response)
    
    # check if request data has extra columns
    if not actual_columns.issubset(necessary_columns):
        extra_columns = actual_columns - necessary_columns
        response = {'error': 'Unrecognized columns provided: {}'.format(extra_columns)}
        return jsonify(response)
    
    # check 'Type' data
    valid_type = ['Person search', 'Person and Vehicle search', 'Vehicle search']
    typ = req['Type']
    if typ not in valid_type:
        response = {'error': 'Invalid value provided for "Type": {}. Allowed values are: {}'.format(typ, valid_type)}
        return jsonify(response)
    
    # check 'Latitude' data
    lat = req['Latitude']
    if lat == lat and not 48 <= lat < 59:
        response = {'error': 'Invalid value provided for "Latitude": {}. Needs to be in [48, 59) or NaN.'.format(lat)}
        return jsonify(response)
    
    # check 'Longitude' data
    long = req['Longitude']
    if long == long and not -10 <= long < 3:
        response = {'error': 'Invalid value provided for "Longitude": {}. Needs to be in [-10, 3) or NaN.'.format(lat)}
        return jsonify(response)
    
    # check 'Gender' data
    valid_gender = ['Male', 'Female', 'Other']
    gend = req['Gender']
    if gend not in valid_gender:
        response = {'error': 'Invalid value provided for "Gender": {}. Allowed values are: {}'.format(gend, valid_gender)}
        return jsonify(response)
    
    # check 'Age range' data
    valid_age = ['18-24', 'over 34', '10-17', '25-34', 'under 10']
    age = req['Age range']
    if age not in valid_age:
        response = {'error': 'Invalid value provided for "Age range": {}. Allowed values are: {}'.format(age, valid_age)}
        return jsonify(response)
    
    # check 'Officer-defined ethnicity' data
    valid_eth = ['White', 'Black', 'Asian', 'Other', 'Mixed']
    eth = req['Officer-defined ethnicity']
    if eth not in valid_eth:
        response = {'error': 'Invalid value provided for "Officer-defined ethnicity": {}. Allowed values are: {}'.format(eth, valid_eth)}
        return jsonify(response)
    
    # check 'Part of a policing operation' data
    valid_op = [True, False]
    op = req['Part of a policing operation']
    if op not in valid_op and op == op:
        response = {'error': 'Invalid value provided for "Part of a policing operation": {}. Needs to be True/False or NaN.'.format(op, valid_op)}
        return jsonify(response)
    
    # check 'Date' data
    date = req['Date']
    if not type(date) == str:
        response = {'error': 'Invalid value provided for "Date": {}. Needs to be a string.'.format(date)}
        return jsonify(response)
    
    # check 'Legislation' data
    leg = req['Legislation']
    if not type(leg) == str:
        response = {'error': 'Invalid value provided for "Legislation": {}. Needs to be a string.'.format(leg)}
        return jsonify(response)
    
    # check 'Object of search' data
    obj = req['Object of search']
    if not type(obj) == str:
        response = {'error': 'Invalid value provided for "Object of search": {}. Needs to be a string.'.format(obj)}
        return jsonify(response)
    
    # check 'station' data
    stat = req['station']
    if not type(stat) == str:
        response = {'error': 'Invalid value provided for "station": {}. Needs to be a string.'.format(stat)}
        return jsonify(response)
    
    # NOTE: The fields 'Date', Legislation', 'Object of search', 'station' are only checked for string type
    
    

    ### PREDICTION ###
    
    # initialise input data
    keys = ['Type',
            'Date',
            'Part of a policing operation',
            'Latitude',
            'Longitude',
            'Gender',
            'Age range',
            'Officer-defined ethnicity',
            'Legislation',
            'Object of search',
            'station']
    X = pd.DataFrame([[req[key] for key in keys]], index=[req['observation_id']], columns=columns).astype(dtypes)
    
    # create output
    authorisation = src.evaluate.authorise_search(pipeline, X)[0]
    if authorisation:
        predicted_outcome = 'true'
    else:
        predicted_outcome = 'false'
    response = {'outcome': predicted_outcome}

    # store
    p = Prediction(
        observation_id=req['observation_id'],
        predicted_outcome=predicted_outcome,
        request=req,
        )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(req['observation_id'])
        response["error"] = error_msg
        # print(error_msg)
        DB.rollback()

    return jsonify(response)


# SEARCH_RESULT ENDPOINT

@app.route('/search_result/', methods=['POST'])
def update():
    raw = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == raw['observation_id'])
        p.true_outcome = raw['outcome']
        p.save()
        
        response = {'observation_id': p.observation_id,
                    'outcome': p.true_outcome,
                    'predicted_outcome': p.predicted_outcome,
                    }
        return jsonify(response)
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(raw['observation_id'])
        response = {'error': error_msg}
        return jsonify(response)


# @app.route('/list-db-contents')
# def list_db_contents():
#     return jsonify([
#         model_to_dict(obs) for obs in Prediction.select()
#     ])

# run
if __name__ == "__main__":
    app.run(debug=True)