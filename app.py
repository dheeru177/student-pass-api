from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run()
