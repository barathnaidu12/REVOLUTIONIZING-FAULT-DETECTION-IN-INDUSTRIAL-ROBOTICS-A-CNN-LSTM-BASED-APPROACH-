from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained Gradient Boosting model
with open('gradient_boosting_model.pkl', 'rb') as f:
    gb_model = pickle.load(f)

# Define class labels
class_labels = {
    0: 'No Failure',
    1: 'Failure'
}

# Predict maintenance function
def predict_maintenance(input_data):
    # Make predictions using the Gradient Boosting model
    gb_prediction = gb_model.predict(input_data)
    gb_prediction_label = [class_labels[pred] for pred in gb_prediction]
    return gb_prediction_label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the form
        tool_wear = float(request.form['tool_wear'])
        power = float(request.form['power'])
        temp_diff = float(request.form['temp_diff'])
        type_H = int(request.form['type_H'])
        type_L = int(request.form['type_L'])
        type_M = int(request.form['type_M'])

        # Convert input data to DataFrame
        input_data = pd.DataFrame({
            'Tool wear': [tool_wear],
            'Power': [power],
            'temp_diff': [temp_diff],
            'Type_H': [type_H],
            'Type_L': [type_L],
            'Type_M': [type_M]
        })

        # Make predictions
        predicted_outcome = predict_maintenance(input_data)

        return render_template('result.html', prediction=predicted_outcome[0])

if __name__ == '__main__':
    app.run(debug=True)
