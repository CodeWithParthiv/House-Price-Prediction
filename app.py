from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np



app = Flask(__name__)

# Load the model and preprocessors
model = joblib.load('house_price_model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoders = joblib.load('label_encoders.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract features
        features = {
            'area': float(data['area']),
            'bedrooms': int(data['bedrooms']),
            'bathrooms': int(data['bathrooms']),
            'stories': int(data['stories']),
            'mainroad': data['mainroad'],
            'guestroom': data['guestroom'],
            'basement': data['basement'],
            'hotwaterheating': data['hotwaterheating'],
            'airconditioning': data['airconditioning'],
            'parking': int(data['parking']),
            'prefarea': data['prefarea'],
            'furnishingstatus': data['furnishingstatus']
        }
        
        # Convert categorical variables
        for column in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                      'airconditioning', 'prefarea', 'furnishingstatus']:
            features[column] = label_encoders[column].transform([features[column]])[0]
        
        # Create feature array
        feature_array = np.array([[
            features['area'], features['bedrooms'], features['bathrooms'],
            features['stories'], features['mainroad'], features['guestroom'],
            features['basement'], features['hotwaterheating'],
            features['airconditioning'], features['parking'],
            features['prefarea'], features['furnishingstatus']
        ]])
        
        # Scale features
        scaled_features = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        return jsonify({
            'predicted_price': float(prediction),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(debug=True) 
