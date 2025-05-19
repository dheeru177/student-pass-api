from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load your trained model (make sure the .pkl file is in the same directory)
model = joblib.load('random_forest_model.pkl')

# List of expected features after one-hot encoding (from your training)
EXPECTED_FEATURES = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime',
    'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2',
    'school_GP', 'school_MS', 'sex_F', 'sex_M', 'address_R', 'address_U',
    'famsize_GT3', 'famsize_LE3', 'Pstatus_A', 'Pstatus_T',
    'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher',
    'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher',
    'reason_course', 'reason_home', 'reason_other', 'reason_reputation',
    'guardian_father', 'guardian_mother', 'guardian_other',
    'schoolsup_no', 'schoolsup_yes', 'famsup_no', 'famsup_yes',
    'paid_no', 'paid_yes', 'activities_no', 'activities_yes',
    'nursery_no', 'nursery_yes', 'higher_no', 'higher_yes',
    'internet_no', 'internet_yes', 'romantic_no', 'romantic_yes'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()

        # Convert input into DataFrame (one row)
        df = pd.DataFrame([input_data])

        # One-hot encode categorical features (your training used get_dummies)
        df = pd.get_dummies(df)

        # Add missing columns with default 0
        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns to match training data
        df = df[EXPECTED_FEATURES]

        prediction = model.predict(df)[0]

        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
