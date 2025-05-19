from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load your trained model
model = joblib.load('random_forest_model.pkl')

# Expected features after one-hot encoding (make sure this matches your training)
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

        # Optional: preprocess input_data keys to match your model feature names
        # Example: 'internet' yes/no to 'internet_yes'/'internet_no' etc.
        # Here’s a quick manual transform for the yes/no fields from your PHP form:
        for col_prefix in ['internet', 'romantic']:
            val = input_data.get(col_prefix, '').lower()
            input_data[col_prefix + '_yes'] = 1 if val == 'yes' else 0
            input_data[col_prefix + '_no'] = 1 if val == 'no' else 0
            input_data.pop(col_prefix, None)

        # Similarly, convert categorical features to one-hot format with keys matching EXPECTED_FEATURES
        # For example, school: 'GP' or 'MS' -> 'school_GP' or 'school_MS'
        if 'school' in input_data:
            school_val = input_data.pop('school')
            input_data['school_GP'] = 1 if school_val == 'GP' else 0
            input_data['school_MS'] = 1 if school_val == 'MS' else 0

        if 'sex' in input_data:
            sex_val = input_data.pop('sex')
            input_data['sex_F'] = 1 if sex_val == 'F' else 0
            input_data['sex_M'] = 1 if sex_val == 'M' else 0

        # Jobs
        for parent in ['Mjob', 'Fjob']:
            val = input_data.pop(parent, None)
            for job_option in ['at_home', 'health', 'other', 'services', 'teacher']:
                key = f"{parent}_{job_option}"
                input_data[key] = 1 if val == job_option else 0

        # Create DataFrame from input
        df = pd.DataFrame([input_data])

        # Add missing expected columns with 0
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
