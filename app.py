from flask import Flask, jsonify, request
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    json_ = request.json
    query_df = pd.DataFrame(json_)
    
    # Preprocess the data (e.g., one-hot encoding if needed)
    query = query_df
    
    # Load the saved classifier
    classifier = joblib.load('classifier.pkl')
    
    # Make predictions
    prediction = classifier.predict(query)
    
    # Return the predictions as JSON
    return jsonify({'prediction': list(prediction)})

if __name__ == '__main__':
    app.run(port=8080)
