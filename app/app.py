import os
from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import pandas as pd
import pickle
import csv
from utils import data_validation

# Define paths relative to the script location
model_path = os.path.join('data', 'model.pkl')
features_metadata_path = os.path.join('data', 'features_metadata.pkl')

# Load the model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load feature names from pickle file
with open(features_metadata_path, 'rb') as file:
    features_metadata = pickle.load(file)

features_dtypes_tuple = [(feature, metadata['dtype']) for feature, metadata in features_metadata.items()]

# Flask-APP
app = Flask(__name__, template_folder='../templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({'features_dtypes': features_dtypes_tuple})

@app.route('/predict', methods=['POST'])
def make_prediction():
    try:
        data = request.json
        user_feature_values, warnings = data_validation(data, features_metadata)
        X_new = pd.DataFrame([user_feature_values], columns=features_metadata.keys())
        Y_new_hat = model.predict(X_new)
        output = np.round(Y_new_hat[0], 2)
        return jsonify({'predicted_price': output, 'warnings': warnings})
    except ValueError as ve:
        return jsonify({'error': str(ve)})
    except Exception as e:
        return jsonify({'error': str(e)})

# Helper function to check allowed file extensions
def allowed_file(filename):
    allowed_files_types = {'csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_files_types

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        file_path = os.path.join(os.path.dirname(__file__), '..', 'uploads/user-data', file.filename)
        file.save(file_path)
        predictions, warnings = process_batch_file(file_path)
        
        # Create a CSV file for the predictions
        predictions_path = os.path.join(os.path.dirname(__file__), '..', 'uploads/predictions', 'predictions.csv')
        with open(predictions_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['index', 'predicted_price'])
            for prediction in predictions:
                writer.writerow([prediction['index'], prediction['prediction']])
                
        return send_file(predictions_path, as_attachment=True, download_name='predictions.csv')
    else:
        return jsonify({"error": "Invalid file type"}), 400

def process_batch_file(file_path):
    df = pd.read_csv(file_path)
    estimates = []
    all_warnings = []
    
    for index, row in df.iterrows():
        user_data = row.to_dict()
        try:
            user_feature_values, row_warnings = data_validation(user_data, features_metadata)
            X_new = pd.DataFrame([user_feature_values], columns=features_metadata.keys())
            prediction = model.predict(X_new)[0]
            estimates.append({
                "index": index,
                "prediction": np.round(prediction, 2)
            })
            all_warnings.extend(row_warnings)
        except ValueError as e:
            all_warnings.append({"index": index, "error": str(e)})
    
    return estimates, all_warnings

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
