from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_path = 'model.pkl'  # Path to your trained model

# Load the model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# Encoding mappings (update these to match your training data encoding)
sex_mapping = {'Male': 0, 'Female': 1}
chest_pain_mapping = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
resting_ecg_mapping = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
exercise_angina_mapping = {'No': 0, 'Yes': 1}
st_slope_mapping = {'Up': 0, 'Flat': 1, 'Down': 2}

@app.route('/')
def home():
    return render_template('index.html')  # Your HTML file for the frontend

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    try:
        age = int(request.form['Age'])
        sex = request.form['Sex']
        chest_pain = request.form['ChestPainType']
        resting_bp = float(request.form['RestingBP'])
        cholesterol = float(request.form['Cholesterol'])
        fasting_bs = int(request.form['FastingBS'])
        resting_ecg = request.form['RestingECG']
        max_hr = int(request.form['MaxHR'])
        exercise_angina = request.form['ExerciseAngina']
        oldpeak = float(request.form['Oldpeak'])
        st_slope = request.form['ST_Slope']

        # Encode categorical variables
        sex_encoded = sex_mapping[sex]
        chest_pain_encoded = chest_pain_mapping[chest_pain]
        resting_ecg_encoded = resting_ecg_mapping[resting_ecg]
        exercise_angina_encoded = exercise_angina_mapping[exercise_angina]
        st_slope_encoded = st_slope_mapping[st_slope]

    except KeyError as e:
        return render_template('index.html', prediction_text=f"Invalid input for categorical variable: {e}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

    # Create a DataFrame with the input features
    input_data = pd.DataFrame([[age, sex_encoded, chest_pain_encoded, resting_bp, cholesterol, fasting_bs,
                                resting_ecg_encoded, max_hr, exercise_angina_encoded, oldpeak, st_slope_encoded]],
                              columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                                       'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                                       'Oldpeak', 'ST_Slope'])

    # Make the prediction using the model
    try:
        prediction = model.predict(input_data)
        # Map prediction result to a human-readable outcome
        output = 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'
    except Exception as e:
        return render_template('index.html', prediction_text=f"Prediction Error: {e}")

    # Return the result in the HTML page
    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
